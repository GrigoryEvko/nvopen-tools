// Function: sub_20422C0
// Address: 0x20422c0
//
__int64 __fastcall sub_20422C0(__int64 a1, __int64 *a2, int a3)
{
  __int64 v3; // rax
  unsigned int v5; // r15d
  __int64 v6; // r8
  __int64 v8; // r8
  __int64 i; // rbx
  unsigned int v10; // r13d
  __int64 v11; // rdi
  __int64 v12; // r9
  __int64 (__fastcall *v13)(__int64, unsigned __int8); // rax
  __int64 v14; // rcx
  __int64 v15; // rbx
  __int64 v16; // r13
  unsigned int *v17; // rsi
  int v18; // eax
  unsigned __int8 v19; // si
  __int64 v20; // rdi
  __int64 v21; // r8
  __int64 (__fastcall *v22)(__int64, unsigned __int8); // rax
  __int64 v23; // rax
  __int64 v25; // rax
  __int64 (__fastcall *v26)(__int64, unsigned __int8); // rax
  __int64 v27; // rax
  int v28; // eax
  __int64 v29; // rax
  __int64 (__fastcall *v30)(__int64, unsigned __int8); // rax
  int v31; // eax
  unsigned int v32; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  unsigned int v34; // [rsp+10h] [rbp-40h]
  __int64 v35; // [rsp+18h] [rbp-38h]
  int v36; // [rsp+18h] [rbp-38h]
  int v37; // [rsp+18h] [rbp-38h]

  if ( a2 )
  {
    v3 = *a2;
    v5 = 0;
    if ( *a2 && *(__int16 *)(v3 + 24) < 0 )
    {
      v6 = *(unsigned int *)(v3 + 60);
      if ( (_DWORD)v6 )
      {
        v8 = 16 * v6;
        for ( i = 0; v8 != i; i += 16 )
        {
          v10 = *(unsigned __int8 *)(*(_QWORD *)(v3 + 40) + i);
          if ( !(_BYTE)v10 )
            continue;
          v11 = *(_QWORD *)(a1 + 136);
          v12 = (unsigned __int8)v10;
          if ( !*(_QWORD *)(v11 + 8LL * (unsigned __int8)v10 + 120) )
            continue;
          v13 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v11 + 288LL);
          if ( v13 == sub_1D45FB0 )
            goto LABEL_9;
          v32 = a3;
          v35 = v8;
          v25 = v13(v11, v10);
          v8 = v35;
          a3 = v32;
          v12 = (unsigned __int8)v10;
          if ( !v25 )
            goto LABEL_10;
          v11 = *(_QWORD *)(a1 + 136);
          v26 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v11 + 288LL);
          if ( v26 == sub_1D45FB0 )
          {
LABEL_9:
            if ( a3 == *(unsigned __int16 *)(**(_QWORD **)(v11 + 8 * v12 + 120) + 24LL) )
              goto LABEL_28;
          }
          else
          {
            v27 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, __int64 (__fastcall *)(__int64, unsigned __int8), __int64, _QWORD))v26)(
                    v11,
                    v10,
                    v32,
                    sub_1D45FB0,
                    v35,
                    (unsigned __int8)v10);
            a3 = v32;
            v8 = v35;
            if ( v32 == *(unsigned __int16 *)(*(_QWORD *)v27 + 24LL) )
            {
LABEL_28:
              v33 = v8;
              v36 = a3;
              v28 = sub_2041EC0(a1, (__int64)a2, a3);
              v8 = v33;
              a3 = v36;
              v5 += v28;
              v3 = *a2;
              continue;
            }
          }
LABEL_10:
          v3 = *a2;
        }
      }
      v14 = *(unsigned int *)(v3 + 56);
      if ( (_DWORD)v14 )
      {
        v15 = 0;
        v16 = 40 * v14;
        while ( 1 )
        {
          v17 = (unsigned int *)(v15 + *(_QWORD *)(v3 + 32));
          v18 = *(unsigned __int16 *)(*(_QWORD *)v17 + 24LL);
          if ( v18 != 32 && v18 != 10 )
          {
            v19 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v17 + 40LL) + 16LL * v17[2]);
            if ( v19 )
            {
              v20 = *(_QWORD *)(a1 + 136);
              v21 = v19;
              if ( *(_QWORD *)(v20 + 8LL * v19 + 120) )
              {
                v22 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v20 + 288LL);
                if ( v22 == sub_1D45FB0 )
                  goto LABEL_20;
                v34 = a3;
                v29 = ((__int64 (*)(void))v22)();
                a3 = v34;
                v21 = v19;
                if ( !v29 )
                  goto LABEL_23;
                v20 = *(_QWORD *)(a1 + 136);
                v30 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v20 + 288LL);
                if ( v30 == sub_1D45FB0 )
                {
LABEL_20:
                  v23 = *(_QWORD *)(v20 + 8 * v21 + 120);
                }
                else
                {
                  v23 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, __int64 (__fastcall *)(__int64, unsigned __int8), _QWORD))v30)(
                          v20,
                          v19,
                          v34,
                          sub_1D45FB0,
                          v19);
                  a3 = v34;
                }
                if ( a3 == *(unsigned __int16 *)(*(_QWORD *)v23 + 24LL) )
                {
                  v37 = a3;
                  v31 = sub_2041DA0(a1, (__int64)a2, a3);
                  a3 = v37;
                  v5 -= v31;
                }
              }
            }
          }
LABEL_23:
          v15 += 40;
          if ( v16 == v15 )
            return v5;
          v3 = *a2;
        }
      }
    }
  }
  else
  {
    return 0;
  }
  return v5;
}
