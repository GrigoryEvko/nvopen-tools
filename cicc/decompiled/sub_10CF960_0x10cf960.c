// Function: sub_10CF960
// Address: 0x10cf960
//
__int64 __fastcall sub_10CF960(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6, char a7)
{
  unsigned int v9; // r15d
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int **v19; // r12
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdx
  int v24; // r13d
  __int64 v25; // r13
  unsigned int *v26; // r12
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // rax
  int v32[8]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v33; // [rsp+40h] [rbp-70h]
  _BYTE v34[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v35; // [rsp+70h] [rbp-40h]

  v9 = (a6 == 0) + 28;
  v10 = sub_10CF840(a1, a2, a3, a5, a6, 0);
  if ( v10 )
  {
    v11 = a1[2].m128i_i64[0];
    v12 = v10;
    if ( a7 )
    {
      v13 = *(_QWORD *)(a4 + 8);
      v35 = 257;
      if ( v9 == 29 )
      {
        v17 = sub_AD62B0(v13);
        return sub_B36550((unsigned int **)v11, v12, v17, a4, (__int64)v34, 0);
      }
      else
      {
        v14 = sub_AD6530(v13, a2);
        return sub_B36550((unsigned int **)v11, v12, a4, v14, (__int64)v34, 0);
      }
    }
    else
    {
      v33 = 257;
      v15 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(v11 + 80) + 16LL))(
              *(_QWORD *)(v11 + 80),
              v9,
              v10,
              a4);
      if ( !v15 )
      {
        v35 = 257;
        v15 = sub_B504D0(v9, v12, a4, (__int64)v34, 0, 0);
        if ( (unsigned __int8)sub_920620(v15) )
        {
          v23 = *(_QWORD *)(v11 + 96);
          v24 = *(_DWORD *)(v11 + 104);
          if ( v23 )
            sub_B99FD0(v15, 3u, v23);
          sub_B45150(v15, v24);
        }
        (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(v11 + 88) + 16LL))(
          *(_QWORD *)(v11 + 88),
          v15,
          v32,
          *(_QWORD *)(v11 + 56),
          *(_QWORD *)(v11 + 64));
        v25 = *(_QWORD *)v11 + 16LL * *(unsigned int *)(v11 + 8);
        if ( *(_QWORD *)v11 != v25 )
        {
          v26 = *(unsigned int **)v11;
          do
          {
            v27 = *((_QWORD *)v26 + 1);
            v28 = *v26;
            v26 += 4;
            sub_B99FD0(v15, v28, v27);
          }
          while ( (unsigned int *)v25 != v26 );
        }
      }
    }
  }
  else
  {
    v18 = sub_10CF840(a1, a2, a4, a5, a6, 0);
    v15 = v18;
    if ( v18 )
    {
      v19 = (unsigned int **)a1[2].m128i_i64[0];
      v35 = 257;
      if ( a7 )
      {
        if ( v9 == 29 )
        {
          v29 = sub_AD62B0(*(_QWORD *)(v18 + 8));
          v22 = v15;
          v21 = v29;
        }
        else
        {
          v20 = sub_AD6530(*(_QWORD *)(v18 + 8), a2);
          v21 = v15;
          v22 = v20;
        }
        return sub_B36550(v19, a3, v21, v22, (__int64)v34, 0);
      }
      else
      {
        return sub_10BBE20((__int64 *)v19, v9, a3, v18, v32[0], 0, (__int64)v34, 0);
      }
    }
  }
  return v15;
}
