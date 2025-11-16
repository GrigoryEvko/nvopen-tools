// Function: sub_F154C0
// Address: 0xf154c0
//
__int64 __fastcall sub_F154C0(__int64 a1, unsigned int a2, char a3, __int64 a4, __int64 a5)
{
  char v6; // dl
  _BYTE *v10; // rax
  __int64 *v11; // rdi
  __int64 v12; // rax
  char v13; // dl
  __int64 v14; // rax
  __int64 v15; // r13
  char v16; // dl
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r12
  __int64 v20; // rax
  bool v21; // zf
  __int64 v22; // r15
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v26; // rdi
  __int64 v27; // rdx
  int v28; // r13d
  unsigned int *v29; // rbx
  __int64 v30; // r13
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // rax
  __int64 v34; // [rsp+0h] [rbp-B0h]
  __int64 v36; // [rsp+8h] [rbp-A8h]
  _QWORD v37[4]; // [rsp+1Fh] [rbp-91h] BYREF
  __int16 v38; // [rsp+40h] [rbp-70h]
  _BYTE v39[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v40; // [rsp+70h] [rbp-40h]

  v6 = 0;
  v10 = *(_BYTE **)a1;
  v11 = *(__int64 **)(a1 + 16);
  LOBYTE(v37[0]) = *v10;
  v12 = *(_QWORD *)(a5 + 16);
  if ( v12 )
    v6 = *(_QWORD *)(v12 + 8) == 0;
  if ( !sub_F13D80(v11, a5, v6, 0, v37, **(_DWORD **)(a1 + 8)) )
    return 0;
  v13 = 0;
  v14 = *(_QWORD *)(a4 + 16);
  if ( v14 )
    v13 = *(_QWORD *)(v14 + 8) == 0;
  v15 = sub_F13D80(*(__int64 **)(a1 + 16), a4, v13, **(__int64 ***)(a1 + 24), v37, **(_DWORD **)(a1 + 8));
  if ( !v15 )
    return 0;
  v16 = 0;
  v17 = *(_QWORD *)(a5 + 16);
  if ( v17 )
    v16 = *(_QWORD *)(v17 + 8) == 0;
  v18 = a5;
  v19 = 1;
  v20 = sub_F13D80(*(__int64 **)(a1 + 16), v18, v16, **(__int64 ***)(a1 + 24), v37, **(_DWORD **)(a1 + 8));
  v21 = a3 == 0;
  **(_BYTE **)a1 = v37[0];
  v22 = **(_QWORD **)(a1 + 24);
  if ( v21 )
  {
    if ( v22 )
    {
      v26 = *(_QWORD *)(v22 + 80);
      v34 = v20;
      v38 = 257;
      v19 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v26 + 16LL))(v26, a2, v15, v20);
      if ( !v19 )
      {
        v40 = 257;
        v19 = sub_B504D0(a2, v15, v34, (__int64)v39, 0, 0);
        if ( (unsigned __int8)sub_920620(v19) )
        {
          v27 = *(_QWORD *)(v22 + 96);
          v28 = *(_DWORD *)(v22 + 104);
          if ( v27 )
            sub_B99FD0(v19, 3u, v27);
          sub_B45150(v19, v28);
        }
        (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(v22 + 88) + 16LL))(
          *(_QWORD *)(v22 + 88),
          v19,
          (char *)v37 + 1,
          *(_QWORD *)(v22 + 56),
          *(_QWORD *)(v22 + 64));
        v29 = *(unsigned int **)v22;
        v30 = *(_QWORD *)v22 + 16LL * *(unsigned int *)(v22 + 8);
        if ( *(_QWORD *)v22 != v30 )
        {
          do
          {
            v31 = *((_QWORD *)v29 + 1);
            v32 = *v29;
            v29 += 4;
            sub_B99FD0(v19, v32, v31);
          }
          while ( (unsigned int *)v30 != v29 );
        }
      }
    }
  }
  else if ( v22 )
  {
    v21 = a2 == 29;
    v36 = v20;
    v40 = 257;
    v23 = *(_QWORD *)(v20 + 8);
    if ( v21 )
    {
      v24 = sub_AD62B0(v23);
      return sub_B36550((unsigned int **)v22, v15, v24, v36, (__int64)v39, 0);
    }
    else
    {
      v33 = sub_AD6530(v23, v18);
      return sub_B36550((unsigned int **)v22, v15, v36, v33, (__int64)v39, 0);
    }
  }
  return v19;
}
