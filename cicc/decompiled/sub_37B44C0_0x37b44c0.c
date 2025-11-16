// Function: sub_37B44C0
// Address: 0x37b44c0
//
__int64 __fastcall sub_37B44C0(__int64 a1, __int64 *a2, int a3)
{
  unsigned int v3; // r8d
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v9; // rcx
  __int64 i; // rbx
  unsigned int v11; // r13d
  __int64 v12; // rdi
  __int64 v13; // r10
  __int64 (__fastcall *v14)(__int64, unsigned __int16); // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // r13
  unsigned int *v18; // rcx
  int v19; // eax
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rcx
  __int64 (__fastcall *v23)(__int64, unsigned __int16); // rax
  __int64 v24; // rax
  __int64 v26; // rax
  __int64 (__fastcall *v27)(__int64, unsigned __int16); // rax
  __int64 v28; // rax
  int v29; // eax
  __int64 v30; // rax
  __int64 (__fastcall *v31)(__int64, unsigned __int16); // rax
  unsigned int v32; // [rsp-44h] [rbp-44h]
  unsigned int v33; // [rsp-44h] [rbp-44h]
  unsigned int v34; // [rsp-44h] [rbp-44h]
  __int64 v35; // [rsp-40h] [rbp-40h]
  __int64 v36; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return 0;
  v3 = 0;
  v5 = *a2;
  if ( *a2 && *(int *)(v5 + 24) < 0 )
  {
    v6 = *(unsigned int *)(v5 + 68);
    if ( (_DWORD)v6 )
    {
      v9 = 16 * v6;
      for ( i = 0; v9 != i; i += 16 )
      {
        v11 = *(unsigned __int16 *)(*(_QWORD *)(v5 + 48) + i);
        if ( !(_WORD)v11 )
          continue;
        v12 = *(_QWORD *)(a1 + 136);
        v13 = (unsigned __int16)v11;
        if ( !*(_QWORD *)(v12 + 8LL * (unsigned __int16)v11 + 112) )
          continue;
        v14 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v12 + 552LL);
        if ( v14 == sub_2EC09E0 )
          goto LABEL_9;
        v32 = v3;
        v35 = v9;
        v26 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v14)(v12, v11, 0);
        v9 = v35;
        v3 = v32;
        v13 = (unsigned __int16)v11;
        if ( !v26 )
          goto LABEL_10;
        v12 = *(_QWORD *)(a1 + 136);
        v27 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v12 + 552LL);
        if ( v27 == sub_2EC09E0 )
        {
LABEL_9:
          if ( a3 == *(unsigned __int16 *)(**(_QWORD **)(v12 + 8 * v13 + 112) + 24LL) )
            goto LABEL_28;
        }
        else
        {
          v28 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v27)(v12, v11, 0);
          v3 = v32;
          v9 = v35;
          if ( a3 == *(unsigned __int16 *)(*(_QWORD *)v28 + 24LL) )
          {
LABEL_28:
            v33 = v3;
            v36 = v9;
            v29 = sub_37B40C0(a1, (__int64)a2, a3);
            v9 = v36;
            v3 = v29 + v33;
            v5 = *a2;
            continue;
          }
        }
LABEL_10:
        v5 = *a2;
      }
    }
    v15 = *(unsigned int *)(v5 + 64);
    if ( (_DWORD)v15 )
    {
      v16 = 0;
      v17 = 40 * v15;
      while ( 1 )
      {
        v18 = (unsigned int *)(v16 + *(_QWORD *)(v5 + 40));
        v19 = *(_DWORD *)(*(_QWORD *)v18 + 24LL);
        if ( v19 != 35 && v19 != 11 )
        {
          v20 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v18 + 48LL) + 16LL * v18[2]);
          if ( (_WORD)v20 )
          {
            v21 = *(_QWORD *)(a1 + 136);
            v22 = (unsigned __int16)v20;
            if ( *(_QWORD *)(v21 + 8LL * (unsigned __int16)v20 + 112) )
            {
              v23 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v21 + 552LL);
              if ( v23 == sub_2EC09E0 )
                goto LABEL_20;
              v34 = v3;
              v30 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v23)(v21, v20, 0);
              v3 = v34;
              v22 = (unsigned __int16)v20;
              if ( !v30 )
                goto LABEL_23;
              v21 = *(_QWORD *)(a1 + 136);
              v31 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v21 + 552LL);
              if ( v31 == sub_2EC09E0 )
              {
LABEL_20:
                v24 = *(_QWORD *)(v21 + 8 * v22 + 112);
              }
              else
              {
                v24 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, __int64 (__fastcall *)(__int64, unsigned __int16)))v31)(
                        v21,
                        (unsigned int)v20,
                        0,
                        (unsigned __int16)v20,
                        v34,
                        sub_2EC09E0);
                v3 = v34;
              }
              if ( a3 == *(unsigned __int16 *)(*(_QWORD *)v24 + 24LL) )
                v3 -= sub_37B3FA0(a1, (__int64)a2, a3);
            }
          }
        }
LABEL_23:
        v16 += 40;
        if ( v17 == v16 )
          return v3;
        v5 = *a2;
      }
    }
  }
  return v3;
}
