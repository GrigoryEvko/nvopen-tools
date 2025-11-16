// Function: sub_3970600
// Address: 0x3970600
//
__int64 __fastcall sub_3970600(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r15
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // rax
  unsigned int v7; // eax
  __int64 v8; // rdi
  void (*v9)(); // rax
  __int64 v10; // rdx
  unsigned int v11; // r12d
  __int64 *v12; // r13
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // rax
  __int64 v17; // r13
  __int64 *v18; // r14
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 (*v21)(); // rax
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  _BYTE *v26; // r8
  __int64 v27; // rdx
  __int64 v28; // r15
  char v29; // dl
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // rax
  void (__fastcall *v33)(__int64, __int64, unsigned __int64); // r12
  __int64 v34; // rax
  __int64 v35; // r14
  __int64 *v36; // rsi
  __int64 *v37; // rax
  __int64 *v38; // rcx
  __int64 v39; // r14
  __int64 v40; // rax
  __int64 v41; // [rsp+0h] [rbp-130h]
  __int64 v42; // [rsp+10h] [rbp-120h]
  int v43; // [rsp+1Ch] [rbp-114h]
  __int64 v44; // [rsp+28h] [rbp-108h]
  unsigned __int64 v45; // [rsp+30h] [rbp-100h]
  __int64 v46; // [rsp+38h] [rbp-F8h]
  char v47; // [rsp+42h] [rbp-EEh]
  int v48; // [rsp+44h] [rbp-ECh]
  void (__fastcall *v49)(__int64, __int64, _QWORD); // [rsp+48h] [rbp-E8h]
  __int64 v50; // [rsp+48h] [rbp-E8h]
  __int64 v51; // [rsp+48h] [rbp-E8h]
  void (__fastcall *v52)(__int64, __int64, _QWORD); // [rsp+48h] [rbp-E8h]
  __int64 v53; // [rsp+50h] [rbp-E0h] BYREF
  _BYTE *v54; // [rsp+58h] [rbp-D8h]
  _BYTE *v55; // [rsp+60h] [rbp-D0h]
  __int64 v56; // [rsp+68h] [rbp-C8h]
  int v57; // [rsp+70h] [rbp-C0h]
  _BYTE v58[184]; // [rsp+78h] [rbp-B8h] BYREF

  v46 = sub_1E0A0C0(*(_QWORD *)(a1 + 264));
  result = *(_QWORD *)(a1 + 264);
  v3 = *(_QWORD *)(result + 72);
  if ( !v3 || *(_DWORD *)v3 == 4 || *(_QWORD *)(v3 + 8) == *(_QWORD *)(v3 + 16) )
    return result;
  v4 = *(_QWORD *)result;
  v5 = sub_396DD80(a1);
  v47 = (*(__int64 (__fastcall **)(__int64, bool, __int64))(*(_QWORD *)v5 + 64LL))(v5, *(_DWORD *)v3 == 3, v4);
  if ( v47 != 1 )
  {
    v6 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v5 + 56LL))(v5, v4, *(_QWORD *)(a1 + 232));
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 160LL))(*(_QWORD *)(a1 + 256), v6, 0);
    v7 = sub_1E0A790((_DWORD *)v3, v46);
    if ( !v7 )
    {
      sub_396F480(a1, 0xFFFFFFFF, 0);
      goto LABEL_25;
    }
LABEL_6:
    _BitScanReverse(&v7, v7);
    sub_396F480(a1, 31 - (v7 ^ 0x1F), 0);
    if ( v47 )
      goto LABEL_7;
LABEL_25:
    v10 = *(_QWORD *)(v3 + 8);
    result = 0xAAAAAAAAAAAAAAABLL * ((*(_QWORD *)(v3 + 16) - v10) >> 3);
    v48 = result;
    if ( !(_DWORD)result )
      return result;
    goto LABEL_10;
  }
  v7 = sub_1E0A790((_DWORD *)v3, v46);
  if ( v7 )
    goto LABEL_6;
  sub_396F480(a1, 0xFFFFFFFF, 0);
LABEL_7:
  v8 = *(_QWORD *)(a1 + 256);
  v9 = *(void (**)())(*(_QWORD *)v8 + 208LL);
  if ( v9 != nullsub_583 )
    ((void (__fastcall *)(__int64, __int64))v9)(v8, 3);
  v10 = *(_QWORD *)(v3 + 8);
  v48 = -1431655765 * ((*(_QWORD *)(v3 + 16) - v10) >> 3);
  if ( !v48 )
  {
LABEL_27:
    v19 = *(_QWORD *)(a1 + 256);
    result = *(_QWORD *)(*(_QWORD *)v19 + 208LL);
    if ( (void (*)())result != nullsub_583 )
      return ((__int64 (__fastcall *)(__int64, __int64))result)(v19, 4);
    return result;
  }
LABEL_10:
  v11 = 0;
  while ( 1 )
  {
    v12 = (__int64 *)(v10 + 24LL * v11);
    result = v12[1];
    if ( *v12 != result )
    {
      if ( *(_DWORD *)v3 == 3 && *(_BYTE *)(*(_QWORD *)(a1 + 240) + 296LL) )
      {
        v56 = 16;
        v57 = 0;
        v55 = v58;
        v54 = v58;
        v20 = *(_QWORD *)(a1 + 264);
        v53 = 0;
        v21 = *(__int64 (**)())(**(_QWORD **)(v20 + 16) + 56LL);
        if ( v21 == sub_1D12D20 )
          BUG();
        v22 = v21();
        v23 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v22 + 1040LL))(
                v22,
                *(_QWORD *)(a1 + 264),
                v11,
                *(_QWORD *)(a1 + 248));
        v24 = (unsigned __int64)v55;
        v42 = v23;
        v25 = *v12;
        v26 = v54;
        v27 = (v12[1] - *v12) >> 3;
        if ( (_DWORD)v27 )
        {
          v41 = v3;
          v43 = v11;
          v28 = 0;
          v51 = 8LL * (unsigned int)(v27 - 1);
          while ( 1 )
          {
            v35 = *(_QWORD *)(v25 + v28);
            if ( (_BYTE *)v24 != v26 )
              goto LABEL_33;
            v36 = (__int64 *)(v24 + 8LL * HIDWORD(v56));
            if ( v36 == (__int64 *)v24 )
              break;
            v37 = (__int64 *)v24;
            v38 = 0;
            while ( v35 != *v37 )
            {
              if ( *v37 == -2 )
                v38 = v37;
              if ( v36 == ++v37 )
              {
                if ( !v38 )
                  goto LABEL_51;
                *v38 = v35;
                --v57;
                ++v53;
                goto LABEL_34;
              }
            }
LABEL_35:
            if ( v51 == v28 )
            {
              v11 = v43;
              v3 = v41;
              goto LABEL_47;
            }
            v25 = *v12;
            v28 += 8;
          }
LABEL_51:
          if ( HIDWORD(v56) >= (unsigned int)v56 )
          {
LABEL_33:
            sub_16CCBA0((__int64)&v53, v35);
            v24 = (unsigned __int64)v55;
            v26 = v54;
            if ( !v29 )
              goto LABEL_35;
          }
          else
          {
            ++HIDWORD(v56);
            *v36 = v35;
            ++v53;
          }
LABEL_34:
          v30 = *(_QWORD *)(a1 + 248);
          v31 = sub_1DD5A70(v35);
          v32 = sub_38CF310(v31, 0, v30, 0);
          v44 = *(_QWORD *)(a1 + 256);
          v33 = *(void (__fastcall **)(__int64, __int64, unsigned __int64))(*(_QWORD *)v44 + 240LL);
          v45 = sub_38CB1F0(17, v32, v42, *(_QWORD *)(a1 + 248), 0);
          v34 = sub_396FFC0(a1, v43, *(_DWORD *)(v35 + 48));
          v33(v44, v34, v45);
          v24 = (unsigned __int64)v55;
          v26 = v54;
          goto LABEL_35;
        }
LABEL_47:
        if ( (_BYTE *)v24 != v26 )
          _libc_free(v24);
      }
      if ( v47 != 1 && *(_DWORD *)(v46 + 16) == 2 )
      {
        v39 = *(_QWORD *)(a1 + 256);
        v52 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v39 + 176LL);
        v40 = sub_396FFA0(a1, v11, 1);
        v52(v39, v40, 0);
      }
      v13 = *(_QWORD *)(a1 + 256);
      v49 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v13 + 176LL);
      v14 = sub_396FFA0(a1, v11, 0);
      v49(v13, v14, 0);
      v15 = *v12;
      result = (v12[1] - *v12) >> 3;
      if ( (_DWORD)result )
      {
        v50 = 8LL * (unsigned int)(result - 1);
        v16 = v12;
        v17 = 0;
        v18 = v16;
        while ( 1 )
        {
          result = sub_39703E0((_QWORD *)a1, (_DWORD *)v3, *(_QWORD *)(v15 + v17), v11);
          if ( v17 == v50 )
            break;
          v15 = *v18;
          v17 += 8;
        }
      }
    }
    if ( ++v11 == v48 )
      break;
    v10 = *(_QWORD *)(v3 + 8);
  }
  if ( v47 )
    goto LABEL_27;
  return result;
}
