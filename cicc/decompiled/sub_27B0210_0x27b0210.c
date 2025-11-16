// Function: sub_27B0210
// Address: 0x27b0210
//
__int64 __fastcall sub_27B0210(__int64 a1, const void **a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // r13d
  unsigned int v8; // r12d
  __int64 v10; // rcx
  __int64 v11; // rdx
  int v12; // eax
  int v13; // ecx
  __int64 v14; // r11
  __int64 v15; // r12
  unsigned int v16; // ebx
  size_t v17; // r15
  __int64 v18; // r13
  char v19; // al
  __int64 v20; // r11
  int v21; // eax
  __int64 v22; // rdx
  size_t v23; // rdx
  int v24; // eax
  char v25; // al
  unsigned int v26; // ebx
  __int64 v27; // [rsp+0h] [rbp-140h]
  __int64 v28; // [rsp+0h] [rbp-140h]
  __int64 v29; // [rsp+0h] [rbp-140h]
  int v30; // [rsp+8h] [rbp-138h]
  int v31; // [rsp+8h] [rbp-138h]
  int v32; // [rsp+8h] [rbp-138h]
  int v33; // [rsp+Ch] [rbp-134h]
  __int64 *v34; // [rsp+30h] [rbp-110h]
  __int64 v35; // [rsp+48h] [rbp-F8h]
  unsigned __int64 v36[2]; // [rsp+50h] [rbp-F0h] BYREF
  _BYTE v37[32]; // [rsp+60h] [rbp-E0h] BYREF
  unsigned __int64 v38[2]; // [rsp+80h] [rbp-C0h] BYREF
  _BYTE v39[32]; // [rsp+90h] [rbp-B0h] BYREF
  unsigned __int64 v40[2]; // [rsp+B0h] [rbp-90h] BYREF
  _BYTE v41[32]; // [rsp+C0h] [rbp-80h] BYREF
  unsigned __int64 v42[2]; // [rsp+E0h] [rbp-60h] BYREF
  _BYTE v43[80]; // [rsp+F0h] [rbp-50h] BYREF

  v7 = *(_DWORD *)(a1 + 24);
  if ( v7 )
  {
    v35 = *(_QWORD *)(a1 + 8);
    if ( !byte_4FFC580 && (unsigned int)sub_2207590((__int64)&byte_4FFC580) )
    {
      qword_4FFC5B0 = 0;
      qword_4FFC5A0 = (__int64)&qword_4FFC5B0;
      qword_4FFC5D0 = (__int64)algn_4FFC5E0;
      qword_4FFC5D8 = 0x400000000LL;
      qword_4FFC5A8 = 0x400000001LL;
      __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC5A0, &qword_4A427C0);
      sub_2207640((__int64)&byte_4FFC580);
    }
    v36[0] = (unsigned __int64)v37;
    v36[1] = 0x400000000LL;
    if ( (_DWORD)qword_4FFC5A8 )
      sub_27ABF90((__int64)v36, (__int64)&qword_4FFC5A0, (__int64)a3, a4, a5, a6);
    v10 = (unsigned int)qword_4FFC5D8;
    v38[0] = (unsigned __int64)v39;
    v38[1] = 0x400000000LL;
    if ( (_DWORD)qword_4FFC5D8 )
      sub_27AC1D0((__int64)v38, (__int64)&qword_4FFC5D0, (__int64)a3, (unsigned int)qword_4FFC5D8, a5, a6);
    if ( !byte_4FFC508 && (unsigned int)sub_2207590((__int64)&byte_4FFC508) )
    {
      qword_4FFC530 = 1;
      qword_4FFC520 = (__int64)&qword_4FFC530;
      qword_4FFC550 = (__int64)algn_4FFC560;
      qword_4FFC558 = 0x400000000LL;
      qword_4FFC528 = 0x400000001LL;
      __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC520, &qword_4A427C0);
      sub_2207640((__int64)&byte_4FFC508);
    }
    v11 = (unsigned int)qword_4FFC528;
    v40[0] = (unsigned __int64)v41;
    v40[1] = 0x400000000LL;
    if ( (_DWORD)qword_4FFC528 )
      sub_27ABF90((__int64)v40, (__int64)&qword_4FFC520, (unsigned int)qword_4FFC528, v10, a5, a6);
    v42[1] = 0x400000000LL;
    v42[0] = (unsigned __int64)v43;
    if ( (_DWORD)qword_4FFC558 )
      sub_27AC1D0((__int64)v42, (__int64)&qword_4FFC550, v11, v10, a5, a6);
    v12 = sub_27B0000(*a2, (__int64)*a2 + 8 * *((unsigned int *)a2 + 2));
    v13 = v7 - 1;
    v14 = 0;
    v34 = a3;
    v33 = 1;
    v15 = *((unsigned int *)a2 + 2);
    v16 = (v7 - 1) & v12;
    v17 = 8 * v15;
    while ( 1 )
    {
      v18 = v35 + 96LL * v16;
      if ( v15 == *(_DWORD *)(v18 + 8) )
      {
        if ( !v17 || (v31 = v13, v28 = v14, v21 = memcmp(*a2, *(const void **)v18, v17), v13 = v31, v14 = v28, !v21) )
        {
          v22 = *((unsigned int *)a2 + 14);
          if ( v22 == *(_DWORD *)(v18 + 56) )
          {
            v23 = 8 * v22;
            if ( !v23
              || (v29 = v14, v32 = v13, v24 = memcmp(a2[6], *(const void **)(v18 + 48), v23), v13 = v32, v14 = v29, !v24) )
            {
              *v34 = v18;
              v8 = 1;
              goto LABEL_26;
            }
          }
        }
      }
      v27 = v14;
      v30 = v13;
      v19 = sub_27ABCC0(v18, (__int64)v36);
      v20 = v27;
      if ( v19 )
        break;
      v25 = sub_27ABCC0(v18, (__int64)v40);
      v14 = v27;
      v13 = v30;
      if ( !v27 && v25 )
        v14 = v35 + 96LL * v16;
      v26 = v33 + v16;
      ++v33;
      v16 = v30 & v26;
    }
    if ( !v27 )
      v20 = v35 + 96LL * v16;
    *v34 = v20;
    v8 = 0;
LABEL_26:
    if ( (_BYTE *)v42[0] != v43 )
      _libc_free(v42[0]);
    if ( (_BYTE *)v40[0] != v41 )
      _libc_free(v40[0]);
    if ( (_BYTE *)v38[0] != v39 )
      _libc_free(v38[0]);
    if ( (_BYTE *)v36[0] != v37 )
      _libc_free(v36[0]);
  }
  else
  {
    *a3 = 0;
    return 0;
  }
  return v8;
}
