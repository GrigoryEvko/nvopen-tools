// Function: sub_1F01F70
// Address: 0x1f01f70
//
void __fastcall sub_1F01F70(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rdx
  _QWORD *v7; // rdi
  unsigned int v8; // r15d
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 v11; // r8
  unsigned __int64 v12; // r12
  __int64 v13; // [rsp+8h] [rbp-88h]
  _QWORD *v14; // [rsp+10h] [rbp-80h] BYREF
  __int64 v15; // [rsp+18h] [rbp-78h]
  _QWORD v16[14]; // [rsp+20h] [rbp-70h] BYREF

  v6 = 1;
  v15 = 0x800000001LL;
  v14 = v16;
  v16[0] = a1;
  v7 = v16;
  do
  {
    v8 = 0;
    v9 = v7[(unsigned int)v6 - 1];
    v10 = *(_QWORD *)(v9 + 112);
    v11 = v10 + 16LL * *(unsigned int *)(v9 + 120);
    if ( v10 == v11 )
      goto LABEL_16;
    a4 = 1;
    do
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v12 + 236) & 2) != 0 )
          break;
        if ( HIDWORD(v15) <= (unsigned int)v6 )
        {
          a2 = v16;
          v13 = v11;
          sub_16CD150((__int64)&v14, v16, 0, 8, v11, a6);
          v6 = (unsigned int)v15;
          v11 = v13;
        }
        v10 += 16;
        a4 = 0;
        v14[v6] = v12;
        v6 = (unsigned int)(v15 + 1);
        LODWORD(v15) = v15 + 1;
        if ( v11 == v10 )
          goto LABEL_11;
      }
      if ( v8 < *(_DWORD *)(v12 + 244) + *(_DWORD *)(v10 + 12) )
        v8 = *(_DWORD *)(v12 + 244) + *(_DWORD *)(v10 + 12);
      v10 += 16;
    }
    while ( v11 != v10 );
LABEL_11:
    if ( (_BYTE)a4 )
    {
LABEL_16:
      v6 = (unsigned int)(v6 - 1);
      LODWORD(v15) = v6;
      if ( *(_DWORD *)(v9 + 244) != v8 )
      {
        sub_1F01900(v9, (__int64)a2, v6, a4, v11, a6);
        *(_DWORD *)(v9 + 244) = v8;
        v6 = (unsigned int)v15;
      }
      *(_BYTE *)(v9 + 236) |= 2u;
    }
    v7 = v14;
  }
  while ( (_DWORD)v6 );
  if ( v14 != v16 )
    _libc_free((unsigned __int64)v14);
}
