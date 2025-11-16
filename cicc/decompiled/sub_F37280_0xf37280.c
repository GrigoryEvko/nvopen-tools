// Function: sub_F37280
// Address: 0xf37280
//
void __fastcall sub_F37280(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v7; // r12
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // r8
  unsigned int *v12; // rax
  int v13; // ecx
  unsigned __int64 *v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rbx
  __int64 v17; // r8
  unsigned __int64 *v18; // rax
  int v19; // ecx
  unsigned __int64 *v20; // rdx
  _QWORD *v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdx
  bool v25; // zf
  __int64 v26; // rdx
  __int64 v27; // rbx
  unsigned __int64 v28; // r15
  __int64 v29; // [rsp-138h] [rbp-138h]
  __int64 v30; // [rsp-130h] [rbp-130h]
  __int64 v31; // [rsp-110h] [rbp-110h]
  __int64 v32; // [rsp-108h] [rbp-108h]
  __int64 v33; // [rsp-108h] [rbp-108h]
  __int64 v34; // [rsp-100h] [rbp-100h]
  __int64 v35; // [rsp-D8h] [rbp-D8h] BYREF
  unsigned __int64 v36; // [rsp-D0h] [rbp-D0h] BYREF
  unsigned __int64 *v37; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v38; // [rsp-C0h] [rbp-C0h]
  _BYTE v39[32]; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v40; // [rsp-98h] [rbp-98h]
  unsigned __int64 v41; // [rsp-90h] [rbp-90h]
  __int16 v42; // [rsp-88h] [rbp-88h]
  __int64 v43; // [rsp-80h] [rbp-80h]
  void **v44; // [rsp-78h] [rbp-78h]
  void **v45; // [rsp-70h] [rbp-70h]
  __int64 v46; // [rsp-68h] [rbp-68h]
  int v47; // [rsp-60h] [rbp-60h]
  __int16 v48; // [rsp-5Ch] [rbp-5Ch]
  char v49; // [rsp-5Ah] [rbp-5Ah]
  __int64 v50; // [rsp-58h] [rbp-58h]
  __int64 v51; // [rsp-50h] [rbp-50h]
  void *v52; // [rsp-48h] [rbp-48h] BYREF
  void *v53; // [rsp-40h] [rbp-40h] BYREF

  if ( !a2 )
    BUG();
  v7 = (__int64 *)a2;
  v8 = *(_QWORD *)(a2 + 16);
  v34 = a2 - 24;
  v9 = sub_AA48A0(v8);
  v40 = v8;
  v37 = (unsigned __int64 *)v39;
  v38 = 0x200000000LL;
  v43 = v9;
  v44 = &v52;
  v45 = &v53;
  v52 = &unk_49DA100;
  v46 = 0;
  v47 = 0;
  v53 = &unk_49DA0B0;
  v48 = 512;
  v49 = 7;
  v50 = 0;
  v51 = 0;
  v41 = a2;
  v42 = a3;
  if ( a2 != v8 + 48 )
  {
    v36 = *(_QWORD *)sub_B46C60(v34);
    if ( v36 && (sub_B96E90((__int64)&v36, v36, 1), (v11 = v36) != 0) )
    {
      a2 = (unsigned int)v38;
      v12 = (unsigned int *)v37;
      v13 = v38;
      v14 = &v37[2 * (unsigned int)v38];
      if ( v37 != v14 )
      {
        while ( 1 )
        {
          v10 = *v12;
          if ( !(_DWORD)v10 )
            break;
          v12 += 4;
          if ( v14 == (unsigned __int64 *)v12 )
            goto LABEL_41;
        }
        *((_QWORD *)v12 + 1) = v36;
        goto LABEL_10;
      }
LABEL_41:
      if ( (unsigned int)v38 >= (unsigned __int64)HIDWORD(v38) )
      {
        a2 = (unsigned int)v38 + 1LL;
        if ( HIDWORD(v38) < a2 )
        {
          a2 = (unsigned __int64)v39;
          v33 = v36;
          sub_C8D5F0((__int64)&v37, v39, (unsigned int)v38 + 1LL, 0x10u, v36, v10);
          v11 = v33;
          v14 = &v37[2 * (unsigned int)v38];
        }
        *v14 = 0;
        v14[1] = v11;
        v11 = v36;
        LODWORD(v38) = v38 + 1;
      }
      else
      {
        if ( v14 )
        {
          *(_DWORD *)v14 = 0;
          v14[1] = v11;
          v13 = v38;
          v11 = v36;
        }
        LODWORD(v38) = v13 + 1;
      }
    }
    else
    {
      a2 = 0;
      sub_93FB40((__int64)&v37, 0);
      v11 = v36;
    }
    if ( v11 )
    {
LABEL_10:
      a2 = v11;
      sub_B91220((__int64)&v36, v11);
    }
  }
  if ( *(_BYTE *)a1 == 17 )
  {
    v15 = *(_QWORD **)(a1 + 24);
    if ( *(_DWORD *)(a1 + 32) > 0x40u )
      v15 = (_QWORD *)*v15;
    if ( (_DWORD)v15 )
    {
      v31 = (unsigned int)v15;
      v32 = *(_QWORD *)(a1 + 8);
      v16 = 0;
      while ( 1 )
      {
        v40 = v7[2];
        v41 = (unsigned __int64)v7;
        v42 = a3;
        v36 = *(_QWORD *)sub_B46C60(v34);
        if ( !v36 )
          break;
        sub_B96E90((__int64)&v36, v36, 1);
        v17 = v36;
        if ( !v36 )
          break;
        v18 = v37;
        v19 = v38;
        v20 = &v37[2 * (unsigned int)v38];
        if ( v37 == v20 )
        {
LABEL_32:
          if ( (unsigned int)v38 >= (unsigned __int64)HIDWORD(v38) )
          {
            v28 = v30 & 0xFFFFFFFF00000000LL;
            v30 &= 0xFFFFFFFF00000000LL;
            if ( HIDWORD(v38) < (unsigned __int64)(unsigned int)v38 + 1 )
            {
              v29 = v36;
              sub_C8D5F0((__int64)&v37, v39, (unsigned int)v38 + 1LL, 0x10u, v36, (unsigned int)v38 + 1LL);
              v17 = v29;
              v20 = &v37[2 * (unsigned int)v38];
            }
            *v20 = v28;
            v20[1] = v17;
            v17 = v36;
            LODWORD(v38) = v38 + 1;
          }
          else
          {
            if ( v20 )
            {
              *(_DWORD *)v20 = 0;
              v20[1] = v17;
              v19 = v38;
              v17 = v36;
            }
            LODWORD(v38) = v19 + 1;
          }
LABEL_30:
          if ( !v17 )
            goto LABEL_24;
          goto LABEL_23;
        }
        while ( *(_DWORD *)v18 )
        {
          v18 += 2;
          if ( v20 == v18 )
            goto LABEL_32;
        }
        v18[1] = v36;
LABEL_23:
        sub_B91220((__int64)&v36, v17);
LABEL_24:
        v21 = (_QWORD *)v32;
        v22 = v16;
        v23 = sub_AD64C0(v32, v16, 0);
        v25 = *(_QWORD *)(a4 + 16) == 0;
        v36 = v23;
        if ( v25 )
          goto LABEL_51;
        a2 = (unsigned __int64)&v37;
        ++v16;
        (*(void (__fastcall **)(__int64, unsigned __int64 **, unsigned __int64 *))(a4 + 24))(a4, &v37, &v36);
        if ( v31 == v16 )
          goto LABEL_26;
      }
      sub_93FB40((__int64)&v37, 0);
      v17 = v36;
      goto LABEL_30;
    }
  }
  else
  {
    v21 = &v37;
    v22 = sub_F369B0(a1, v7, a3);
    v27 = v26;
    sub_D5F1F0((__int64)&v37, v22);
    v25 = *(_QWORD *)(a4 + 16) == 0;
    v35 = v27;
    if ( v25 )
LABEL_51:
      sub_4263D6(v21, v22, v24);
    a2 = (unsigned __int64)&v37;
    (*(void (__fastcall **)(__int64, unsigned __int64 **, __int64 *))(a4 + 24))(a4, &v37, &v35);
  }
LABEL_26:
  nullsub_61();
  v52 = &unk_49DA100;
  nullsub_63();
  if ( v37 != (unsigned __int64 *)v39 )
    _libc_free(v37, a2);
}
