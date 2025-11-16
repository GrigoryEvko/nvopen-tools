// Function: sub_2B21190
// Address: 0x2b21190
//
__int64 __fastcall sub_2B21190(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, char a5, __int64 a6)
{
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r10
  __int64 v16; // rdx
  unsigned int v17; // eax
  unsigned int v18; // r11d
  __int64 v19; // rbx
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rdx
  int v24; // r12d
  __int64 v25; // r12
  __int64 v26; // rdx
  unsigned int v27; // esi
  char v28; // al
  __int64 v29; // r9
  __int64 v30; // rdx
  int v31; // ebx
  __int64 v32; // rbx
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // [rsp+0h] [rbp-B0h]
  __int64 v36; // [rsp+0h] [rbp-B0h]
  __int64 v37; // [rsp+0h] [rbp-B0h]
  __int64 v38; // [rsp+8h] [rbp-A8h]
  __int64 v39; // [rsp+8h] [rbp-A8h]
  __int64 v41; // [rsp+18h] [rbp-98h]
  unsigned int v42; // [rsp+18h] [rbp-98h]
  int v43; // [rsp+18h] [rbp-98h]
  __int64 v44; // [rsp+18h] [rbp-98h]
  __int64 v45; // [rsp+18h] [rbp-98h]
  __int64 v46; // [rsp+18h] [rbp-98h]
  __int64 v47; // [rsp+18h] [rbp-98h]
  __int64 v48; // [rsp+18h] [rbp-98h]
  __int64 v49; // [rsp+18h] [rbp-98h]
  _DWORD v50[8]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v51; // [rsp+40h] [rbp-70h]
  _BYTE v52[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v53; // [rsp+70h] [rbp-40h]

  v10 = *(_QWORD *)(a3 + 8);
  v35 = v10;
  v11 = v10;
  if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
    v11 = **(_QWORD **)(v10 + 16);
  v41 = v11;
  v38 = v10;
  v12 = sub_BCB2A0(*(_QWORD **)(a2 + 72));
  v13 = v41;
  v14 = *(unsigned int *)(a1 + 1576);
  if ( v41 != v12 || (_DWORD)v14 != 1 )
    goto LABEL_4;
  v13 = a6;
  if ( (unsigned int)*(unsigned __int8 *)(a6 + 8) - 17 <= 1 )
    v13 = **(_QWORD **)(a6 + 16);
  if ( (unsigned int)*(unsigned __int8 *)(v38 + 8) - 17 <= 1 )
    v35 = **(_QWORD **)(v38 + 16);
  if ( v35 != v13 )
  {
    v51 = 257;
    v21 = sub_BCD140(*(_QWORD **)(a2 + 72), *(_DWORD *)(v38 + 32));
    if ( v21 == *(_QWORD *)(a3 + 8) )
    {
      v22 = a3;
    }
    else
    {
      v44 = v21;
      v22 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a2 + 80) + 120LL))(
              *(_QWORD *)(a2 + 80),
              49,
              a3,
              v21);
      if ( !v22 )
      {
        v53 = 257;
        v46 = sub_B51D30(49, a3, v44, (__int64)v52, 0, 0);
        v28 = sub_920620(v46);
        v29 = v46;
        if ( v28 )
        {
          v30 = *(_QWORD *)(a2 + 96);
          v31 = *(_DWORD *)(a2 + 104);
          if ( v30 )
          {
            sub_B99FD0(v46, 3u, v30);
            v29 = v46;
          }
          v47 = v29;
          sub_B45150(v29, v31);
          v29 = v47;
        }
        v48 = v29;
        (*(void (__fastcall **)(_QWORD, __int64, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
          *(_QWORD *)(a2 + 88),
          v29,
          v50,
          *(_QWORD *)(a2 + 56),
          *(_QWORD *)(a2 + 64));
        v22 = v48;
        v37 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v37 )
        {
          v32 = *(_QWORD *)a2;
          do
          {
            v33 = *(_QWORD *)(v32 + 8);
            v34 = *(_DWORD *)v32;
            v49 = v22;
            v32 += 16;
            sub_B99FD0(v22, v34, v33);
            v22 = v49;
          }
          while ( v37 != v32 );
        }
      }
    }
    v53 = 257;
    v50[1] = 0;
    v15 = sub_B33BC0(a2, 0x42u, v22, v50[0], (__int64)v52);
  }
  else
  {
LABEL_4:
    v15 = sub_F70250(a2, a3, v14, v13, v14);
  }
  v16 = a6;
  if ( (unsigned int)*(unsigned __int8 *)(a6 + 8) - 17 <= 1 )
    v16 = **(_QWORD **)(a6 + 16);
  if ( *(_QWORD *)(v15 + 8) != v16 )
  {
    v36 = v15;
    v51 = 257;
    v39 = *(_QWORD *)(v15 + 8);
    v42 = sub_BCB060(v39);
    v17 = sub_BCB060(a6);
    v18 = 39 - ((a5 == 0) - 1);
    if ( v42 > v17 )
      v18 = 38;
    if ( a6 == v39 )
    {
      v19 = v36;
    }
    else
    {
      v43 = v18;
      v19 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a2 + 80) + 120LL))(
              *(_QWORD *)(a2 + 80),
              v18,
              v36,
              a6);
      if ( !v19 )
      {
        v53 = 257;
        v19 = sub_B51D30(v43, v36, a6, (__int64)v52, 0, 0);
        if ( (unsigned __int8)sub_920620(v19) )
        {
          v23 = *(_QWORD *)(a2 + 96);
          v24 = *(_DWORD *)(a2 + 104);
          if ( v23 )
            sub_B99FD0(v19, 3u, v23);
          sub_B45150(v19, v24);
        }
        (*(void (__fastcall **)(_QWORD, __int64, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
          *(_QWORD *)(a2 + 88),
          v19,
          v50,
          *(_QWORD *)(a2 + 56),
          *(_QWORD *)(a2 + 64));
        v45 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v45 )
        {
          v25 = *(_QWORD *)a2;
          do
          {
            v26 = *(_QWORD *)(v25 + 8);
            v27 = *(_DWORD *)v25;
            v25 += 16;
            sub_B99FD0(v19, v27, v26);
          }
          while ( v45 != v25 );
        }
      }
    }
    v15 = v19;
  }
  if ( a4 > 1 )
    return sub_2B1CED0(a1, v15, a2, a4);
  else
    return v15;
}
