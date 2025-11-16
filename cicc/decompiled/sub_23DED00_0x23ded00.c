// Function: sub_23DED00
// Address: 0x23ded00
//
void __fastcall sub_23DED00(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int64 v7; // r10
  __int64 *v9; // r13
  __int64 *v11; // r11
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // r13
  __int64 v16; // rcx
  char v17; // dl
  unsigned __int64 v18; // rbx
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 *v23; // r10
  __int64 v24; // r9
  __int64 *v25; // r11
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rbx
  __int64 v31; // r12
  __int64 v32; // r13
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // [rsp+8h] [rbp-108h]
  __int64 *v36; // [rsp+10h] [rbp-100h]
  __int64 *v37; // [rsp+18h] [rbp-F8h]
  __int64 *v38; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v39; // [rsp+18h] [rbp-F8h]
  __int64 *v40; // [rsp+20h] [rbp-F0h]
  __int64 *v41; // [rsp+20h] [rbp-F0h]
  __int64 v42; // [rsp+20h] [rbp-F0h]
  __int64 *v43; // [rsp+20h] [rbp-F0h]
  __int64 *v44; // [rsp+30h] [rbp-E0h]
  __int64 v45; // [rsp+30h] [rbp-E0h]
  __int64 v46; // [rsp+40h] [rbp-D0h]
  __int64 v47; // [rsp+40h] [rbp-D0h]
  char v49[32]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v50; // [rsp+70h] [rbp-A0h]
  char v51[32]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v52; // [rsp+A0h] [rbp-70h]
  _QWORD v53[4]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v54; // [rsp+D0h] [rbp-40h]

  v7 = a5;
  v9 = a1;
  if ( a4 >= a5 )
  {
    v13 = a4;
  }
  else
  {
    v11 = a1;
    v12 = a4 + 1;
    v13 = a4;
    v14 = a4;
    do
    {
      if ( *(_BYTE *)(a2 + v14) && (v16 = *(unsigned __int8 *)(a3 + v14), v11[2 * v16 + 154]) )
      {
        if ( v12 < a5 )
        {
          while ( 1 )
          {
            v17 = *(_BYTE *)(a2 + v12);
            v18 = v12++;
            if ( !v17 || (_BYTE)v16 != *(_BYTE *)(a3 + v12 - 1) )
              break;
            if ( a5 <= v12 )
              goto LABEL_16;
          }
        }
        else
        {
LABEL_16:
          v18 = v12;
        }
        v46 = v18 - v14;
        if ( v18 - v14 >= *(unsigned int *)(v11[1] + 1100) )
        {
          v37 = &v11[2 * v16];
          v44 = v11;
          sub_23DE110(v11, a2, a3, v13, v14, (__int64 *)a6, a7);
          v52 = 257;
          v19 = v44[2];
          v20 = v44[58];
          v40 = v44;
          v50 = 257;
          v45 = v19;
          v21 = sub_AD64C0(v20, v14, 0);
          v22 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a6 + 80)
                                                                                             + 32LL))(
                  *(_QWORD *)(a6 + 80),
                  13,
                  a7,
                  v21,
                  0,
                  0);
          v23 = v37;
          v24 = v22;
          v25 = v40;
          if ( !v22 )
          {
            v54 = 257;
            v36 = v40;
            v42 = sub_B504D0(13, a7, v21, (__int64)v53, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a6 + 88) + 16LL))(
              *(_QWORD *)(a6 + 88),
              v42,
              v49,
              *(_QWORD *)(a6 + 56),
              *(_QWORD *)(a6 + 64));
            v24 = v42;
            v23 = v37;
            v25 = v36;
            if ( *(_QWORD *)a6 != *(_QWORD *)a6 + 16LL * *(unsigned int *)(a6 + 8) )
            {
              v43 = v37;
              v39 = v18;
              v30 = v24;
              v35 = a2;
              v31 = *(_QWORD *)a6;
              v32 = *(_QWORD *)a6 + 16LL * *(unsigned int *)(a6 + 8);
              do
              {
                v33 = *(_QWORD *)(v31 + 8);
                v34 = *(_DWORD *)v31;
                v31 += 16;
                sub_B99FD0(v30, v34, v33);
              }
              while ( v32 != v31 );
              v24 = v30;
              v23 = v43;
              v18 = v39;
              v25 = v36;
              a2 = v35;
            }
          }
          v26 = v25[58];
          v41 = v25;
          v38 = v23;
          v53[0] = v24;
          v53[1] = sub_AD64C0(v26, v46, 0);
          v27 = sub_921880((unsigned int **)a6, v38[153], v38[154], (int)v53, 2, (__int64)v51, 0);
          v11 = v41;
          if ( *(_BYTE *)(v45 + 8) )
          {
            v29 = *(unsigned int *)(v45 + 24);
            if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(v45 + 28) )
            {
              v47 = v27;
              sub_C8D5F0(v45 + 16, (const void *)(v45 + 32), v29 + 1, 8u, v29 + 1, v28);
              v29 = *(unsigned int *)(v45 + 24);
              v11 = v41;
              v27 = v47;
            }
            *(_QWORD *)(*(_QWORD *)(v45 + 16) + 8 * v29) = v27;
            ++*(_DWORD *)(v45 + 24);
          }
          v14 = v18;
          v13 = v18;
        }
        else
        {
          v14 = v18;
        }
      }
      else
      {
        v14 = v12;
      }
      v12 = v14 + 1;
    }
    while ( a5 > v14 );
    v9 = v11;
    v7 = a5;
  }
  sub_23DE110(v9, a2, a3, v13, v7, (__int64 *)a6, a7);
}
