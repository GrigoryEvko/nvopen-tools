// Function: sub_159D390
// Address: 0x159d390
//
__int64 __fastcall sub_159D390(__int64 a1)
{
  int v2; // r14d
  unsigned int v3; // r14d
  __int64 v4; // rbx
  _BYTE *v5; // rdx
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 *v8; // rdi
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  unsigned int v11; // ebx
  __int64 v13; // [rsp+8h] [rbp-178h]
  int v14; // [rsp+1Ch] [rbp-164h] BYREF
  __int64 v15[4]; // [rsp+20h] [rbp-160h] BYREF
  _BYTE *v16; // [rsp+40h] [rbp-140h] BYREF
  __int64 v17; // [rsp+48h] [rbp-138h]
  _BYTE v18[304]; // [rsp+50h] [rbp-130h] BYREF

  v2 = *(_DWORD *)(a1 + 20);
  v16 = v18;
  v17 = 0x2000000000LL;
  v3 = v2 & 0xFFFFFFF;
  if ( v3 )
  {
    v4 = 1;
    v5 = v18;
    v6 = *(_QWORD *)(a1 - 24LL * v3);
    v7 = 0;
    while ( 1 )
    {
      *(_QWORD *)&v5[8 * v7] = v6;
      v7 = (unsigned int)(v17 + 1);
      LODWORD(v17) = v17 + 1;
      if ( v3 == (_DWORD)v4 )
        break;
      v6 = *(_QWORD *)(a1 + 24 * (v4 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
      if ( HIDWORD(v17) <= (unsigned int)v7 )
      {
        v13 = *(_QWORD *)(a1 + 24 * (v4 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
        sub_16CD150(&v16, v18, 0, 8);
        v7 = (unsigned int)v17;
        v6 = v13;
      }
      v5 = v16;
      ++v4;
    }
    v8 = (__int64 *)v16;
    v9 = &v16[8 * v7];
  }
  else
  {
    v9 = v18;
    v7 = 0;
    v8 = (__int64 *)v18;
  }
  v10 = *(_QWORD *)a1;
  v15[1] = (__int64)v8;
  v15[2] = v7;
  v15[0] = v10;
  v14 = sub_1597240(v8, (__int64)v9);
  v11 = sub_1597ED0(v15, &v14);
  if ( v16 != v18 )
    _libc_free((unsigned __int64)v16);
  return v11;
}
