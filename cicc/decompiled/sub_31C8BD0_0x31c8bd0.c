// Function: sub_31C8BD0
// Address: 0x31c8bd0
//
unsigned __int64 __fastcall sub_31C8BD0(__int64 a1, _QWORD *a2)
{
  _BYTE *v2; // rcx
  unsigned __int64 v3; // r12
  char v4; // dl
  unsigned __int64 result; // rax
  __int64 v6; // rdx
  _BYTE *v7; // rdx
  char v8; // si
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdx
  bool v12; // al
  int v13; // edx
  __int64 v14; // rdx
  __int64 v15; // r9
  _QWORD *v16; // r13
  __int64 v17; // r8
  _QWORD *v18; // rax
  int v19; // edx
  _BYTE *v20; // rcx
  __int64 v21; // r10
  __int64 v22; // r13
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 (__fastcall *v26)(__int64, unsigned __int64, __int64); // rax
  unsigned int *v27; // rbx
  unsigned int *v28; // r14
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v31; // rdi
  unsigned int *v32; // rbx
  unsigned int *v33; // r14
  __int64 v34; // rdx
  unsigned int v35; // esi
  int v36; // [rsp+4h] [rbp-13Ch]
  int v37; // [rsp+8h] [rbp-138h]
  _BYTE v38[32]; // [rsp+10h] [rbp-130h] BYREF
  __int16 v39; // [rsp+30h] [rbp-110h]
  _BYTE *v40; // [rsp+40h] [rbp-100h] BYREF
  __int64 v41; // [rsp+48h] [rbp-F8h]
  _BYTE v42[16]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v43; // [rsp+60h] [rbp-E0h]
  unsigned int *v44; // [rsp+80h] [rbp-C0h] BYREF
  int v45; // [rsp+88h] [rbp-B8h]
  char v46; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+B8h] [rbp-88h]
  __int64 v48; // [rsp+C0h] [rbp-80h]
  __int64 v49; // [rsp+D0h] [rbp-70h]
  __int64 v50; // [rsp+D8h] [rbp-68h]
  void *v51; // [rsp+100h] [rbp-40h]

  v2 = *(_BYTE **)(a1 + 40);
  v3 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = *(_BYTE *)v3;
  if ( *(_BYTE *)v3 == 61 )
    goto LABEL_2;
  if ( v4 != 63 )
  {
    if ( v4 != 78 )
    {
LABEL_19:
      if ( v2 )
      {
        result = 0;
        goto LABEL_5;
      }
      return *a2 & 0xFFFFFFFFFFFFFFF8LL;
    }
LABEL_2:
    result = *(_QWORD *)(v3 - 32);
    if ( result )
      goto LABEL_3;
LABEL_9:
    if ( v2 )
      goto LABEL_5;
    return *a2 & 0xFFFFFFFFFFFFFFF8LL;
  }
  result = *(_QWORD *)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
  if ( !result )
    goto LABEL_9;
LABEL_3:
  if ( *(_BYTE *)result <= 0x1Cu )
    goto LABEL_19;
  if ( v2 != (_BYTE *)result )
  {
    while ( 1 )
    {
LABEL_5:
      v6 = *(_QWORD *)(result + 16);
      if ( !v6 )
        return result;
      v7 = *(_BYTE **)(v6 + 8);
      if ( v7 )
        return result;
      v8 = *(_BYTE *)v3;
      if ( *(_BYTE *)v3 != 61 )
      {
        if ( v8 == 63 )
        {
          v9 = *(_BYTE **)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
          if ( !v9 )
            goto LABEL_16;
          goto LABEL_14;
        }
        if ( v8 != 78 )
          goto LABEL_16;
      }
      v9 = *(_BYTE **)(v3 - 32);
      if ( !v9 )
        goto LABEL_16;
LABEL_14:
      if ( *v9 >= 0x1Du )
        v7 = v9;
LABEL_16:
      v3 = result;
      if ( v2 == v7 )
      {
        if ( !v2 )
          return result;
        v4 = *(_BYTE *)result;
        break;
      }
      result = (unsigned __int64)v7;
    }
  }
  if ( v4 == 61 )
  {
    result = *(_QWORD *)(a1 + 104);
    if ( *(_QWORD *)(v3 - 32) )
    {
      v10 = *(_QWORD *)(v3 - 24);
      **(_QWORD **)(v3 - 16) = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v3 - 16);
    }
    *(_QWORD *)(v3 - 32) = result;
    if ( result )
    {
      v11 = *(_QWORD *)(result + 16);
      *(_QWORD *)(v3 - 24) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = v3 - 24;
      *(_QWORD *)(v3 - 16) = result + 16;
      *(_QWORD *)(result + 16) = v3 - 32;
      return 0;
    }
  }
  else
  {
    if ( v4 == 78 )
    {
      sub_23D0AB0((__int64)&v44, v3, 0, 0, 0);
      v39 = 257;
      v23 = (__int64 *)sub_BD5C60(v3);
      v24 = sub_BCE3C0(v23, 0);
      v22 = *(_QWORD *)(a1 + 104);
      v25 = v24;
      if ( v24 != *(_QWORD *)(v22 + 8) )
      {
        if ( *(_BYTE *)v22 > 0x15u )
        {
          v31 = *(_QWORD *)(a1 + 104);
          v43 = 257;
          v22 = sub_B52190(v31, v24, (__int64)&v40, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v50 + 16LL))(
            v50,
            v22,
            v38,
            v47,
            v48);
          v32 = v44;
          v33 = &v44[4 * v45];
          if ( v44 != v33 )
          {
            do
            {
              v34 = *((_QWORD *)v32 + 1);
              v35 = *v32;
              v32 += 4;
              sub_B99FD0(v22, v35, v34);
            }
            while ( v33 != v32 );
          }
        }
        else
        {
          v26 = *(__int64 (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v49 + 144LL);
          if ( v26 == sub_B32D70 )
            v22 = sub_ADB060(*(_QWORD *)(a1 + 104), v25);
          else
            v22 = v26(v49, *(_QWORD *)(a1 + 104), v25);
          if ( *(_BYTE *)v22 > 0x1Cu )
          {
            (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v50 + 16LL))(
              v50,
              v22,
              v38,
              v47,
              v48);
            v27 = v44;
            v28 = &v44[4 * v45];
            if ( v44 != v28 )
            {
              do
              {
                v29 = *((_QWORD *)v27 + 1);
                v30 = *v27;
                v27 += 4;
                sub_B99FD0(v22, v30, v29);
              }
              while ( v28 != v27 );
            }
          }
        }
      }
    }
    else
    {
      if ( v4 != 63 )
        BUG();
      sub_23D0AB0((__int64)&v44, v3, 0, 0, 0);
      v12 = sub_B4DE30(v3);
      v39 = 257;
      v13 = *(_DWORD *)(v3 + 4);
      v40 = v42;
      v14 = v13 & 0x7FFFFFF;
      v41 = 0x600000000LL;
      v15 = !v12 ? 0 : 3;
      v16 = (_QWORD *)(v3 + 32 * (1 - v14));
      v17 = (-32 * (1 - v14)) >> 5;
      if ( (unsigned __int64)(-32 * (1 - v14)) > 0xC0 )
      {
        v36 = !v12 ? 0 : 3;
        v37 = (-32 * (1 - v14)) >> 5;
        sub_C8D5F0((__int64)&v40, v42, (-32 * (1 - v14)) >> 5, 8u, v17, v15);
        v20 = v40;
        v19 = v41;
        LODWORD(v17) = v37;
        LODWORD(v15) = v36;
        v18 = &v40[8 * (unsigned int)v41];
      }
      else
      {
        v18 = v42;
        v19 = 0;
        v20 = v42;
      }
      if ( v16 != (_QWORD *)v3 )
      {
        do
        {
          if ( v18 )
            *v18 = *v16;
          v16 += 4;
          ++v18;
        }
        while ( (_QWORD *)v3 != v16 );
        v20 = v40;
        v19 = v41;
      }
      v21 = *(_QWORD *)(a1 + 104);
      LODWORD(v41) = v19 + v17;
      v22 = sub_921130(&v44, *(_QWORD *)(v3 + 72), v21, (_BYTE **)v20, (unsigned int)(v19 + v17), (__int64)v38, v15);
      if ( v40 != v42 )
        _libc_free((unsigned __int64)v40);
    }
    nullsub_61();
    v51 = &unk_49DA100;
    nullsub_63();
    if ( v44 != (unsigned int *)&v46 )
      _libc_free((unsigned __int64)v44);
    sub_BD84D0(v3, v22);
    sub_B43D60((_QWORD *)v3);
    return 0;
  }
  return result;
}
