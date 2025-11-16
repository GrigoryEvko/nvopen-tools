// Function: sub_2EB1A60
// Address: 0x2eb1a60
//
__int64 __fastcall sub_2EB1A60(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r14
  _QWORD *v12; // rbx
  _QWORD *v13; // rax
  _QWORD *v14; // rdi
  _QWORD *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v19; // [rsp+10h] [rbp-130h]
  __int64 v20; // [rsp+28h] [rbp-118h]
  _QWORD *v21; // [rsp+28h] [rbp-118h]
  void *v22; // [rsp+30h] [rbp-110h]
  void *v23; // [rsp+38h] [rbp-108h]
  __int64 v24; // [rsp+40h] [rbp-100h] BYREF
  _QWORD *v25; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v26; // [rsp+50h] [rbp-F0h] BYREF
  _QWORD *v27; // [rsp+58h] [rbp-E8h]
  __int64 v28; // [rsp+60h] [rbp-E0h]
  int v29; // [rsp+68h] [rbp-D8h]
  char v30; // [rsp+6Ch] [rbp-D4h]
  _QWORD v31[2]; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v32; // [rsp+80h] [rbp-C0h] BYREF
  _BYTE *v33; // [rsp+88h] [rbp-B8h]
  __int64 v34; // [rsp+90h] [rbp-B0h]
  int v35; // [rsp+98h] [rbp-A8h]
  char v36; // [rsp+9Ch] [rbp-A4h]
  _BYTE v37[16]; // [rsp+A0h] [rbp-A0h] BYREF
  char v38[8]; // [rsp+B0h] [rbp-90h] BYREF
  unsigned __int64 v39; // [rsp+B8h] [rbp-88h]
  char v40; // [rsp+CCh] [rbp-74h]
  unsigned __int64 v41; // [rsp+E8h] [rbp-58h]
  char v42; // [rsp+FCh] [rbp-44h]

  v6 = a1;
  v20 = *(_QWORD *)(sub_BC1CD0(a4, &unk_50209C0, a3) + 8);
  v26 = 1;
  v7 = *(_QWORD *)(sub_BC1CD0(a4, &qword_4F8A320, a3) + 8);
  v36 = 1;
  v32 = 0;
  v24 = v7;
  v27 = v31;
  v33 = v37;
  v34 = 2;
  v35 = 0;
  v28 = 0x100000002LL;
  v29 = 0;
  v30 = 1;
  v31[0] = &qword_4F82400;
  v23 = (void *)(a1 + 32);
  v22 = (void *)(a1 + 80);
  if ( sub_B2FC80(a3)
    || (*(_BYTE *)(a3 + 32) & 0xF) == 1
    || (v19 = *(_QWORD *)(sub_BC1CD0(a4, &unk_5020008, a3) + 8), !(unsigned __int8)sub_2EB0040(&v24, *a2, v19)) )
  {
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = v23;
    *(_QWORD *)(a1 + 64) = 2;
    *(_QWORD *)(a1 + 56) = v22;
    *(_DWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_BYTE *)(a1 + 28) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  else
  {
    (*(void (__fastcall **)(char *, __int64, __int64, __int64))(*(_QWORD *)*a2 + 16LL))(v38, *a2, v19, v20);
    sub_2EB0C30(v20, v19, (__int64)v38);
    if ( v24 )
    {
      v9 = *(_QWORD *)(v24 + 432);
      v21 = (_QWORD *)(v9 + 32LL * *(unsigned int *)(v24 + 440));
      if ( (_QWORD *)v9 != v21 )
      {
        v11 = *a2;
        v12 = *(_QWORD **)(v24 + 432);
        do
        {
          v25 = 0;
          v13 = (_QWORD *)sub_22077B0(0x10u);
          if ( v13 )
          {
            v13[1] = v19;
            *v13 = &unk_4A29888;
          }
          v14 = v25;
          v25 = v13;
          if ( v14 )
            (*(void (__fastcall **)(_QWORD *))(*v14 + 8LL))(v14);
          v15 = v12;
          v17 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 32LL))(v11);
          if ( (v12[3] & 2) == 0 )
            v15 = (_QWORD *)*v12;
          (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD **, char *))(v12[3] & 0xFFFFFFFFFFFFFFF8LL))(
            v15,
            v17,
            v16,
            &v25,
            v38);
          if ( v25 )
            (*(void (__fastcall **)(_QWORD *))(*v25 + 8LL))(v25);
          v12 += 4;
        }
        while ( v21 != v12 );
        v6 = a1;
      }
    }
    sub_BBADB0((__int64)&v26, (__int64)v38, v9, v10);
    sub_C8CF70(v6, v23, 2, (__int64)v31, (__int64)&v26);
    sub_C8CF70(v6 + 48, v22, 2, (__int64)v37, (__int64)&v32);
    if ( !v42 )
      _libc_free(v41);
    if ( !v40 )
      _libc_free(v39);
  }
  if ( !v36 )
    _libc_free((unsigned __int64)v33);
  if ( !v30 )
    _libc_free((unsigned __int64)v27);
  return v6;
}
