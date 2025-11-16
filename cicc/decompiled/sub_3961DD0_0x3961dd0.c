// Function: sub_3961DD0
// Address: 0x3961dd0
//
__int64 __fastcall sub_3961DD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rax
  __int64 v7; // r10
  __int64 v8; // r12
  __int64 v9; // r15
  _BYTE *v10; // rdi
  char v11; // al
  __int64 v12; // rdx
  _QWORD *v13; // rdi
  __int64 v14; // rax
  __int64 *v15; // r8
  __int64 v16; // rax
  __int64 *v17; // rbx
  __int64 v18; // r13
  __int64 v20; // [rsp+10h] [rbp-90h]
  __int64 *v21; // [rsp+18h] [rbp-88h]
  _QWORD *v22; // [rsp+20h] [rbp-80h] BYREF
  __int64 v23; // [rsp+28h] [rbp-78h]
  _QWORD v24[14]; // [rsp+30h] [rbp-70h] BYREF

  v6 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v7 = *(_QWORD *)(a2 - 8);
    v20 = v7 + v6;
    if ( v7 + v6 != v7 )
      goto LABEL_3;
LABEL_26:
    v9 = 0;
    return v9 + (int)sub_1CCB880(a2);
  }
  v20 = a2;
  v7 = a2 - v6;
  if ( a2 == a2 - v6 )
    goto LABEL_26;
LABEL_3:
  v8 = v7;
  v9 = 0;
  do
  {
    v10 = *(_BYTE **)v8;
    v11 = *(_BYTE *)(*(_QWORD *)v8 + 16LL);
    if ( v11 == 5 )
    {
      v22 = v24;
      ++v9;
      LODWORD(v12) = 1;
      v23 = 0x800000001LL;
      v24[0] = v10;
      v13 = v24;
      do
      {
        v14 = (unsigned int)v12;
        v12 = (unsigned int)(v12 - 1);
        v15 = (__int64 *)v13[v14 - 1];
        LODWORD(v23) = v12;
        v16 = 24LL * (*((_DWORD *)v15 + 5) & 0xFFFFFFF);
        if ( (*((_BYTE *)v15 + 23) & 0x40) != 0 )
        {
          v17 = (__int64 *)*(v15 - 1);
          v15 = &v17[(unsigned __int64)v16 / 8];
        }
        else
        {
          v17 = &v15[v16 / 0xFFFFFFFFFFFFFFF8LL];
        }
        if ( v15 != v17 )
        {
          do
          {
            v18 = *v17;
            if ( *(_BYTE *)(*v17 + 16) == 5 )
            {
              if ( HIDWORD(v23) <= (unsigned int)v12 )
              {
                v21 = v15;
                sub_16CD150((__int64)&v22, v24, 0, 8, (int)v15, a6);
                v12 = (unsigned int)v23;
                v15 = v21;
              }
              ++v9;
              v22[v12] = v18;
              v12 = (unsigned int)(v23 + 1);
              LODWORD(v23) = v23 + 1;
            }
            v17 += 3;
          }
          while ( v15 != v17 );
          v13 = v22;
        }
      }
      while ( (_DWORD)v12 );
      if ( v13 != v24 )
        _libc_free((unsigned __int64)v13);
    }
    else if ( v11 == 17 && (unsigned __int8)sub_3960EF0(v10) )
    {
      v9 += 6;
    }
    v8 += 24;
  }
  while ( v20 != v8 );
  return v9 + (int)sub_1CCB880(a2);
}
