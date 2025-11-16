// Function: sub_2F5AF30
// Address: 0x2f5af30
//
__int64 __fastcall sub_2F5AF30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  void (__fastcall *v5)(__int64 *, __int64, __int64); // rax
  __int64 **v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  void **v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 **v36; // rsi
  __int64 v37; // [rsp+0h] [rbp-72A0h] BYREF
  void **v38; // [rsp+8h] [rbp-7298h]
  void (__fastcall *v39)(__int64 *, __int64 *, __int64); // [rsp+10h] [rbp-7290h]
  __int64 v40; // [rsp+18h] [rbp-7288h]
  _BYTE v41[16]; // [rsp+20h] [rbp-7280h] BYREF
  _BYTE v42[8]; // [rsp+30h] [rbp-7270h] BYREF
  unsigned __int64 v43; // [rsp+38h] [rbp-7268h]
  int v44; // [rsp+44h] [rbp-725Ch]
  int v45; // [rsp+48h] [rbp-7258h]
  char v46; // [rsp+4Ch] [rbp-7254h]
  _BYTE v47[16]; // [rsp+50h] [rbp-7250h] BYREF
  __int64 v48[14]; // [rsp+60h] [rbp-7240h] BYREF
  _QWORD v49[3642]; // [rsp+D0h] [rbp-71D0h] BYREF

  *(_QWORD *)(a3 + 344) &= 0xFFEuLL;
  sub_2F4FBD0(v48, a3, a4);
  v5 = *(void (__fastcall **)(__int64 *, __int64, __int64))(a2 + 16);
  v39 = 0;
  if ( v5 )
  {
    v5(&v37, a2, 2);
    v40 = *(_QWORD *)(a2 + 24);
    v39 = *(void (__fastcall **)(__int64 *, __int64 *, __int64))(a2 + 16);
  }
  sub_2F4F660((__int64)v49, v48, (__int64)&v37);
  if ( v39 )
    v39(&v37, &v37, 3);
  if ( !(unsigned __int8)sub_2F5A640(v49, a3) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_7;
  }
  sub_2EAFFB0((__int64)&v37);
  if ( v44 == v45 )
  {
    if ( BYTE4(v40) )
    {
      v11 = v38;
      v36 = (__int64 **)&v38[HIDWORD(v39)];
      v8 = HIDWORD(v39);
      v7 = (__int64 **)v38;
      if ( v38 != (void **)v36 )
      {
        while ( *v7 != &qword_4F82400 )
        {
          if ( v36 == ++v7 )
          {
LABEL_13:
            while ( *v11 != &unk_4F82408 )
            {
              if ( ++v11 == (void **)v7 )
                goto LABEL_18;
            }
            goto LABEL_14;
          }
        }
        goto LABEL_14;
      }
      goto LABEL_18;
    }
    if ( sub_C8CA60((__int64)&v37, (__int64)&qword_4F82400) )
      goto LABEL_14;
  }
  if ( !BYTE4(v40) )
  {
LABEL_20:
    sub_C8CC70((__int64)&v37, (__int64)&unk_4F82408, (__int64)v7, v8, v9, v10);
    goto LABEL_14;
  }
  v11 = v38;
  v8 = HIDWORD(v39);
  v7 = (__int64 **)&v38[HIDWORD(v39)];
  if ( v7 != (__int64 **)v38 )
    goto LABEL_13;
LABEL_18:
  if ( (unsigned int)v8 >= (unsigned int)v39 )
    goto LABEL_20;
  v8 = (unsigned int)(v8 + 1);
  HIDWORD(v39) = v8;
  *v7 = (__int64 *)&unk_4F82408;
  ++v37;
LABEL_14:
  sub_2F4DA50((__int64)&v37, (__int64)&unk_501EC10, (__int64)v7, v8, v9, v10);
  sub_2F4DA50((__int64)&v37, (__int64)&unk_501EAD0, v12, v13, v14, v15);
  sub_2F4DA50((__int64)&v37, (__int64)&qword_5025C20, v16, v17, v18, v19);
  sub_2F4DA50((__int64)&v37, (__int64)&qword_501E910, v20, v21, v22, v23);
  sub_2F4DA50((__int64)&v37, (__int64)&qword_501EB00, v24, v25, v26, v27);
  sub_2F4DA50((__int64)&v37, (__int64)&qword_502A660, v28, v29, v30, v31);
  sub_2F4DA50((__int64)&v37, (__int64)&qword_501EAF0, v32, v33, v34, v35);
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v41, (__int64)&v37);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v47, (__int64)v42);
  if ( !v46 )
    _libc_free(v43);
  if ( !BYTE4(v40) )
    _libc_free((unsigned __int64)v38);
LABEL_7:
  sub_2F4E350((__int64)v49);
  return a1;
}
