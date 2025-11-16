// Function: sub_120BB20
// Address: 0x120bb20
//
__int64 __fastcall sub_120BB20(__int64 a1, const char *a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  unsigned __int64 v5; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // [rsp+8h] [rbp-C8h]
  _QWORD *v12; // [rsp+20h] [rbp-B0h] BYREF
  size_t v13; // [rsp+28h] [rbp-A8h]
  _QWORD v14[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v15[2]; // [rsp+40h] [rbp-90h] BYREF
  const char *v16; // [rsp+50h] [rbp-80h]
  __int64 v17; // [rsp+58h] [rbp-78h]
  __int16 v18; // [rsp+60h] [rbp-70h]
  _QWORD v19[2]; // [rsp+70h] [rbp-60h] BYREF
  const char *v20; // [rsp+80h] [rbp-50h]
  __int64 v21; // [rsp+88h] [rbp-48h]
  __int16 v22; // [rsp+90h] [rbp-40h]

  v4 = *(unsigned __int8 *)(a4 + 8);
  if ( (_BYTE)v4 )
  {
    v16 = a2;
    v15[0] = "field '";
    v22 = 770;
    v5 = *(_QWORD *)(a1 + 232);
    v18 = 1283;
    v17 = a3;
    v19[0] = v15;
    v20 = "' cannot be specified more than once";
    sub_11FD800(a1 + 176, v5, (__int64)v19, 1);
    return v4;
  }
  v12 = v14;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v8 = *(_QWORD *)(a1 + 232);
  v13 = 0;
  v10 = v8;
  LOBYTE(v14[0]) = 0;
  v4 = sub_120B3D0(a1, (__int64)&v12);
  if ( !(_BYTE)v4 )
  {
    if ( *(_BYTE *)(a4 + 9) )
    {
      if ( !v13 )
      {
        v9 = 0;
LABEL_8:
        *(_BYTE *)(a4 + 8) = 1;
        *(_QWORD *)a4 = v9;
        goto LABEL_9;
      }
LABEL_12:
      v9 = sub_B9B140(*(__int64 **)a1, v12, v13);
      goto LABEL_8;
    }
    if ( v13 )
      goto LABEL_12;
    v21 = a3;
    v22 = 1283;
    v19[0] = "'";
    v4 = 1;
    v15[0] = v19;
    v18 = 770;
    v20 = a2;
    v16 = "' cannot be empty";
    sub_11FD800(a1 + 176, v10, (__int64)v15, 1);
  }
LABEL_9:
  if ( v12 != v14 )
    j_j___libc_free_0(v12, v14[0] + 1LL);
  return v4;
}
