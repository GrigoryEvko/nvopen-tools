// Function: sub_23DE5D0
// Address: 0x23de5d0
//
__int64 __fastcall sub_23DE5D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rbx
  _BYTE *v9; // r15
  __int64 v10; // rsi
  _BYTE *v11; // r14
  __int64 v12; // rdi
  __int64 v13; // r12
  unsigned int *v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rdx
  unsigned int v18; // esi
  unsigned int *v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rdx
  unsigned int v22; // esi
  _BYTE v23[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v24; // [rsp+20h] [rbp-70h]
  _BYTE v25[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v26; // [rsp+50h] [rbp-40h]

  v6 = *(int *)(a1 + 120);
  v7 = *(_QWORD *)(a2 + 8);
  v24 = 257;
  v8 = sub_AD64C0(v7, v6, 0);
  v9 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a3 + 80) + 24LL))(
                  *(_QWORD *)(a3 + 80),
                  26,
                  a2,
                  v8,
                  0);
  if ( !v9 )
  {
    v26 = 257;
    v9 = (_BYTE *)sub_B504D0(26, a2, v8, (__int64)v25, 0, 0);
    (*(void (__fastcall **)(_QWORD, _BYTE *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
      *(_QWORD *)(a3 + 88),
      v9,
      v23,
      *(_QWORD *)(a3 + 56),
      *(_QWORD *)(a3 + 64));
    v15 = *(unsigned int **)a3;
    v16 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
    if ( *(_QWORD *)a3 != v16 )
    {
      do
      {
        v17 = *((_QWORD *)v15 + 1);
        v18 = *v15;
        v15 += 4;
        sub_B99FD0((__int64)v9, v18, v17);
      }
      while ( (unsigned int *)v16 != v15 );
    }
  }
  v10 = *(_QWORD *)(a1 + 128);
  if ( v10 )
  {
    v11 = *(_BYTE **)(a1 + 1016);
    if ( !v11 )
      v11 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a1 + 96), v10, 0);
    if ( *(_BYTE *)(a1 + 136) )
    {
      v12 = *(_QWORD *)(a3 + 80);
      v24 = 257;
      v13 = (*(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE *))(*(_QWORD *)v12 + 16LL))(v12, 29, v9, v11);
      if ( !v13 )
      {
        v26 = 257;
        v13 = sub_B504D0(29, (__int64)v9, (__int64)v11, (__int64)v25, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
          *(_QWORD *)(a3 + 88),
          v13,
          v23,
          *(_QWORD *)(a3 + 56),
          *(_QWORD *)(a3 + 64));
        v19 = *(unsigned int **)a3;
        v20 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
        while ( (unsigned int *)v20 != v19 )
        {
          v21 = *((_QWORD *)v19 + 1);
          v22 = *v19;
          v19 += 4;
          sub_B99FD0(v13, v22, v21);
        }
      }
      return v13;
    }
    else
    {
      v26 = 257;
      return sub_929C50((unsigned int **)a3, v9, v11, (__int64)v25, 0, 0);
    }
  }
  return (__int64)v9;
}
