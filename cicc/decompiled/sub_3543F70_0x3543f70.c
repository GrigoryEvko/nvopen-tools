// Function: sub_3543F70
// Address: 0x3543f70
//
__int64 __fastcall sub_3543F70(
        __int64 a1,
        __int64 a2,
        unsigned int *a3,
        unsigned int *a4,
        unsigned int *a5,
        __int64 *a6)
{
  __int64 *v11; // rdi
  __int64 v12; // rax
  __int64 (*v13)(void); // rdx
  __int64 (*v14)(); // rax
  unsigned int v15; // r8d
  char v17; // al
  int v18; // esi
  unsigned __int64 v19; // rax
  __int64 v20; // rdi
  int v21; // eax
  int v22; // eax
  __int64 v23; // r8
  unsigned __int64 v24; // r11
  __int64 v25; // rdi
  __int64 (*v26)(); // rax
  __int64 v27; // rdi
  __int64 (*v28)(); // rax
  _QWORD *v29; // rsi
  __int64 v30; // r8
  __int64 v31; // rdi
  __int64 (*v32)(); // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  unsigned int v37; // [rsp+8h] [rbp-68h]
  unsigned __int64 v38; // [rsp+10h] [rbp-60h]
  unsigned __int8 v39; // [rsp+10h] [rbp-60h]
  int v40; // [rsp+18h] [rbp-58h]
  unsigned __int64 v41; // [rsp+18h] [rbp-58h]
  __int64 v42; // [rsp+18h] [rbp-58h]
  __int64 v43; // [rsp+20h] [rbp-50h]
  int v44; // [rsp+20h] [rbp-50h]
  unsigned __int64 v45; // [rsp+20h] [rbp-50h]
  __int64 v46; // [rsp+20h] [rbp-50h]
  unsigned int v48; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v49; // [rsp+34h] [rbp-3Ch] BYREF
  int v50; // [rsp+38h] [rbp-38h] BYREF
  unsigned int v51[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v11 = *(__int64 **)(a1 + 16);
  v12 = *v11;
  v13 = *(__int64 (**)(void))(*v11 + 912);
  if ( v13 != sub_2FDC6F0 )
  {
    v17 = v13();
    v15 = 0;
    if ( v17 )
      return v15;
    v11 = *(__int64 **)(a1 + 16);
    v12 = *v11;
  }
  v14 = *(__int64 (**)())(v12 + 824);
  if ( v14 != sub_2FDC6B0 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64 *, __int64, unsigned int *, unsigned int *))v14)(v11, a2, &v48, &v49) )
    {
      v18 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 40LL * v48 + 8);
      v43 = *(_QWORD *)(sub_2E88D60(a2) + 32);
      v19 = sub_2EBEE10(v43, v18);
      v20 = v19;
      if ( v19 )
      {
        v21 = *(unsigned __int16 *)(v19 + 68);
        if ( !v21 || v21 == 68 )
        {
          v22 = sub_353D010(v20, *(_QWORD *)(a2 + 24));
          if ( v22 )
          {
            v44 = v22;
            v24 = sub_2EBEE10(v23, v22);
            if ( a2 != v24 && v24 != 0 )
            {
              v25 = *(_QWORD *)(a1 + 16);
              v26 = *(__int64 (**)())(*(_QWORD *)v25 + 912LL);
              if ( v26 != sub_2FDC6F0 )
              {
                v40 = v44;
                v45 = v24;
                if ( ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v26)(v25, v24) )
                {
                  v27 = *(_QWORD *)(a1 + 16);
                  v50 = 0;
                  v51[0] = 0;
                  v28 = *(__int64 (**)())(*(_QWORD *)v27 + 824LL);
                  if ( v28 != sub_2FDC6B0 )
                  {
                    v37 = v40;
                    v41 = v45;
                    if ( ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, int *, unsigned int *))v28)(
                           v27,
                           v45,
                           &v50,
                           v51) )
                    {
                      v38 = v45;
                      v46 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * v49 + 24);
                      v42 = *(_QWORD *)(*(_QWORD *)(v41 + 32) + 40LL * v51[0] + 24);
                      v29 = sub_2E7B2C0(*(_QWORD **)(a1 + 32), a2);
                      *(_QWORD *)(v29[4] + 40LL * v49 + 24) = v42 + v46;
                      v31 = *(_QWORD *)(a1 + 16);
                      v32 = *(__int64 (**)())(*(_QWORD *)v31 + 1264LL);
                      if ( v32 == sub_2E85460 )
                      {
                        sub_2E790D0(*(_QWORD *)(a1 + 32), (__int64)v29, v42 + v46, v42, v30, v37);
                        return 0;
                      }
                      v39 = ((__int64 (__fastcall *)(__int64, _QWORD *, unsigned __int64))v32)(v31, v29, v38);
                      sub_2E790D0(*(_QWORD *)(a1 + 32), (__int64)v29, v33, v34, v35, v36);
                      v15 = v39;
                      if ( v39 )
                      {
                        *a3 = v48;
                        *a4 = v49;
                        *a5 = v37;
                        *a6 = v42;
                        return v15;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return 0;
}
