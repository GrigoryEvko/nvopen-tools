// Function: sub_2597480
// Address: 0x2597480
//
__int64 __fastcall sub_2597480(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7)
{
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdi
  __int64 (__fastcall *v15)(__int64, __int64 (__fastcall *)(__int64, unsigned __int64), __int64, char); // rax
  unsigned __int8 **v16; // rdx
  __int64 result; // rax
  unsigned __int8 **v18; // rbx
  unsigned __int8 **v19; // r15
  int v20; // ebx
  char v21; // al
  unsigned __int8 *v22; // rax
  __int64 v23; // [rsp-10h] [rbp-80h]
  __int64 v24; // [rsp+0h] [rbp-70h]
  __int64 v25; // [rsp+8h] [rbp-68h]
  _QWORD v26[12]; // [rsp+10h] [rbp-60h] BYREF

  v26[3] = a2;
  v25 = a5;
  v24 = a6;
  v26[4] = a5;
  v26[5] = a6;
  v26[0] = &a7;
  v26[1] = a3;
  v26[2] = a1;
  v9 = sub_250D2C0(a4, 0);
  v11 = sub_252AE70(a2, v9, v10, a1, 1, 0, 1);
  if ( !v11 )
    goto LABEL_8;
  v14 = v11;
  v15 = *(__int64 (__fastcall **)(__int64, __int64 (__fastcall *)(__int64, unsigned __int64), __int64, char))(*(_QWORD *)v11 + 112LL);
  if ( v15 != sub_254E4A0 )
  {
    result = ((__int64 (__fastcall *)(__int64, __int64 (__fastcall *)(__int64, unsigned __int8 *), _QWORD *, __int64, __int64, __int64, __int64, __int64))v15)(
               v14,
               sub_25970A0,
               v26,
               1,
               v12,
               v13,
               v24,
               v25);
LABEL_10:
    if ( (_BYTE)result )
      return result;
LABEL_8:
    v20 = (unsigned __int8)sub_B46420(a3);
    v21 = sub_B46490(a3);
    sub_2561E50(a1, v25, 0x80u, a3, 0, v24, (2 * (v21 != 0)) | v20);
    return v23;
  }
  if ( !*(_BYTE *)(v14 + 97) )
  {
    v22 = (unsigned __int8 *)sub_250D070((_QWORD *)(v14 + 72));
    result = sub_2596E60((__int64)v26, v22);
    goto LABEL_10;
  }
  v16 = *(unsigned __int8 ***)(v14 + 136);
  result = *(unsigned int *)(v14 + 144);
  v18 = &v16[result];
  if ( v16 != v18 )
  {
    v19 = *(unsigned __int8 ***)(v14 + 136);
    while ( 1 )
    {
      result = sub_2596E60((__int64)v26, *v19);
      if ( !(_BYTE)result )
        goto LABEL_8;
      if ( v18 == ++v19 )
        goto LABEL_10;
    }
  }
  return result;
}
