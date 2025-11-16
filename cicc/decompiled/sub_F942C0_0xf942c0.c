// Function: sub_F942C0
// Address: 0xf942c0
//
__int64 __fastcall sub_F942C0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v15; // r9
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // r9
  __int64 v19; // rdx
  int v20; // r12d
  unsigned int *v21; // r12
  __int64 v22; // rbx
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+8h] [rbp-68h]
  __int64 v27; // [rsp+8h] [rbp-68h]
  __int64 v28; // [rsp+8h] [rbp-68h]
  _BYTE v29[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v30; // [rsp+30h] [rbp-40h]

  if ( (unsigned __int8)sub_98EF70(a4, a3) )
  {
    v15 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a1 + 80) + 16LL))(
            *(_QWORD *)(a1 + 80),
            a2,
            a3,
            a4);
    if ( !v15 )
    {
      v30 = 257;
      v25 = sub_B504D0(a2, a3, a4, (__int64)v29, 0, 0);
      v17 = sub_920620(v25);
      v18 = v25;
      if ( v17 )
      {
        v19 = *(_QWORD *)(a1 + 96);
        v20 = *(_DWORD *)(a1 + 104);
        if ( v19 )
        {
          sub_B99FD0(v25, 3u, v19);
          v18 = v25;
        }
        v26 = v18;
        sub_B45150(v18, v20);
        v18 = v26;
      }
      v27 = v18;
      (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
        *(_QWORD *)(a1 + 88),
        v18,
        a5,
        *(_QWORD *)(a1 + 56),
        *(_QWORD *)(a1 + 64));
      v21 = *(unsigned int **)a1;
      v15 = v27;
      v22 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
      if ( *(_QWORD *)a1 != v22 )
      {
        do
        {
          v23 = *((_QWORD *)v21 + 1);
          v24 = *v21;
          v21 += 4;
          v28 = v15;
          sub_B99FD0(v15, v24, v23);
          v15 = v28;
        }
        while ( (unsigned int *)v22 != v21 );
      }
    }
    return v15;
  }
  else
  {
    if ( a2 == 28 )
    {
      v16 = sub_AD6530(*(_QWORD *)(a4 + 8), a3);
      v11 = a5;
      v13 = a4;
      v12 = v16;
    }
    else
    {
      if ( a2 != 29 )
        BUG();
      v10 = sub_AD62B0(*(_QWORD *)(a4 + 8));
      v11 = a5;
      v12 = a4;
      v13 = v10;
    }
    return sub_B36550((unsigned int **)a1, a3, v13, v12, v11, 0);
  }
}
