// Function: sub_12343B0
// Address: 0x12343b0
//
__int64 __fastcall sub_12343B0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, int a6)
{
  int v6; // eax
  int v8; // edx
  int v9; // r9d
  __int64 v11; // rax
  unsigned int v12; // r13d
  __int64 v14; // rsi
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // r13
  unsigned __int64 v19; // rdx
  const char *v20; // r14
  unsigned __int64 v21; // rsi
  bool v22; // zf
  unsigned int v23; // eax
  __int64 v24; // r12
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 v27; // r15
  __int64 *v28; // r12
  int v29; // [rsp+Ch] [rbp-1A4h]
  __int64 v30; // [rsp+10h] [rbp-1A0h]
  __int64 v31; // [rsp+30h] [rbp-180h] BYREF
  __int64 v32; // [rsp+38h] [rbp-178h] BYREF
  unsigned __int64 v33[4]; // [rsp+40h] [rbp-170h] BYREF
  __int16 v34; // [rsp+60h] [rbp-150h]
  const char *v35; // [rsp+70h] [rbp-140h] BYREF
  __int64 v36; // [rsp+78h] [rbp-138h]
  _BYTE v37[304]; // [rsp+80h] [rbp-130h] BYREF

  v8 = *(_DWORD *)(a1 + 240);
  LOBYTE(a6) = v8 != 510;
  LOBYTE(v6) = v8 != 55;
  v9 = v6 & a6;
  LOBYTE(v9) = (v8 != 504) & v9;
  if ( (_BYTE)v9 )
  {
    v21 = *(_QWORD *)(a1 + 232);
    v37[17] = 1;
    v12 = v9;
    v35 = "expected scope value for catchswitch";
    v37[16] = 3;
    sub_11FD800(a1 + 176, v21, (__int64)&v35, 1);
    return v12;
  }
  v11 = sub_BCB190(*(_QWORD **)a1);
  if ( (unsigned __int8)sub_1224B80((__int64 **)a1, v11, &v31, a3)
    || (unsigned __int8)sub_120AFE0(a1, 6, "expected '[' with catchswitch labels") )
  {
    return 1;
  }
  v35 = v37;
  v36 = 0x2000000000LL;
  while ( 1 )
  {
    v14 = (__int64)&v32;
    v33[0] = 0;
    v12 = sub_122FEA0(a1, &v32, v33, a3);
    if ( (_BYTE)v12 )
      goto LABEL_12;
    v17 = (unsigned int)v36;
    v18 = v32;
    v19 = (unsigned int)v36 + 1LL;
    if ( v19 > HIDWORD(v36) )
    {
      sub_C8D5F0((__int64)&v35, v37, v19, 8u, v15, v16);
      v17 = (unsigned int)v36;
    }
    *(_QWORD *)&v35[8 * v17] = v18;
    LODWORD(v36) = v36 + 1;
    if ( *(_DWORD *)(a1 + 240) != 4 )
      break;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
  v14 = 7;
  v12 = sub_120AFE0(a1, 7, "expected ']' after catchswitch labels");
  if ( (_BYTE)v12 || (v14 = 66, v12 = sub_120AFE0(a1, 66, "expected 'unwind' after catchswitch scope"), (_BYTE)v12) )
  {
LABEL_12:
    v20 = v35;
    goto LABEL_13;
  }
  v22 = *(_DWORD *)(a1 + 240) == 56;
  v32 = 0;
  if ( v22 )
  {
    v14 = 57;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    v23 = sub_120AFE0(a1, 57, "expected 'caller' in catchswitch");
    if ( !(_BYTE)v23 )
      goto LABEL_19;
  }
  else
  {
    v14 = (__int64)&v32;
    v33[0] = 0;
    v23 = sub_122FEA0(a1, &v32, v33, a3);
    if ( !(_BYTE)v23 )
    {
LABEL_19:
      v34 = 257;
      v30 = v32;
      v24 = v31;
      v29 = v36;
      v25 = sub_BD2DA0(80);
      v27 = v25;
      if ( v25 )
      {
        v14 = v24;
        sub_B4C2D0(v25, v24, v30, v29, (__int64)v33, v26, 0, 0);
      }
      v28 = (__int64 *)v35;
      v20 = &v35[8 * (unsigned int)v36];
      if ( v35 != v20 )
      {
        do
        {
          v14 = *v28++;
          sub_B4C4B0(v27, v14);
        }
        while ( v20 != (const char *)v28 );
        v20 = v35;
      }
      *a2 = v27;
      goto LABEL_13;
    }
  }
  v20 = v35;
  v12 = v23;
LABEL_13:
  if ( v20 != v37 )
    _libc_free(v20, v14);
  return v12;
}
