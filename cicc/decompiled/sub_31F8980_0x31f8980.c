// Function: sub_31F8980
// Address: 0x31f8980
//
void __fastcall sub_31F8980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // rax
  size_t v10; // r13
  _BYTE *v11; // r14
  _BYTE *v12; // r15
  __int64 v13; // rdi
  void (*v14)(); // rax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // rdi
  void (*v18)(); // rax
  __int64 v19; // [rsp+8h] [rbp-188h]
  _QWORD v20[4]; // [rsp+10h] [rbp-180h] BYREF
  char v21; // [rsp+30h] [rbp-160h]
  char v22; // [rsp+31h] [rbp-15Fh]
  _BYTE *v23; // [rsp+40h] [rbp-150h] BYREF
  size_t v24; // [rsp+48h] [rbp-148h]
  __int64 v25; // [rsp+50h] [rbp-140h]
  _BYTE dest[312]; // [rsp+58h] [rbp-138h] BYREF

  v23 = dest;
  v19 = sub_31F8790(a1, 4353, a3, a4, a5);
  v8 = *(_QWORD *)(a1 + 8);
  v24 = 0;
  v9 = *(_QWORD *)(v8 + 200);
  v25 = 256;
  v10 = *(_QWORD *)(v9 + 1240);
  v11 = *(_BYTE **)(v9 + 1232);
  if ( v10 > 0x100 )
  {
    sub_C8D290((__int64)&v23, dest, *(_QWORD *)(v9 + 1240), 1u, v6, v7);
    memcpy(&v23[v24], v11, v10);
    v10 += v24;
    v24 = v10;
LABEL_15:
    v12 = v23;
    goto LABEL_4;
  }
  if ( !v10 )
  {
LABEL_3:
    v12 = 0;
    v10 = 0;
    goto LABEL_4;
  }
  memcpy(dest, *(const void **)(v9 + 1232), *(_QWORD *)(v9 + 1240));
  v24 = v10;
  if ( v10 != 1 )
    goto LABEL_15;
  if ( *v11 == 45 )
    goto LABEL_3;
  v12 = dest;
LABEL_4:
  v13 = *(_QWORD *)(a1 + 528);
  v14 = *(void (**)())(*(_QWORD *)v13 + 120LL);
  v22 = 1;
  v20[0] = "Signature";
  v21 = 3;
  if ( v14 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v14)(v13, v20, 1);
    v13 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v13 + 536LL))(v13, 0, 4);
  v17 = *(__int64 **)(a1 + 528);
  v18 = *(void (**)())(*v17 + 120);
  v22 = 1;
  v20[0] = "Object name";
  v21 = 3;
  if ( v18 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v18)(v17, v20, 1);
    v17 = *(__int64 **)(a1 + 528);
  }
  sub_31F4F00(v17, v12, v10, 3840, v15, v16);
  sub_31F8930(a1, v19);
  if ( v23 != dest )
    _libc_free((unsigned __int64)v23);
}
