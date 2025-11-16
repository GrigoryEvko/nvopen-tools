// Function: sub_38AEBD0
// Address: 0x38aebd0
//
__int64 __fastcall sub_38AEBD0(__int64 a1, __int64 *a2, __int64 *a3, double a4, double a5, double a6)
{
  int v8; // eax
  __int64 v9; // rax
  unsigned int v10; // r13d
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // rax
  const char *v15; // r14
  unsigned __int64 v16; // rsi
  unsigned int v17; // eax
  __int64 *v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r15
  __int64 *v25; // r12
  __int64 v26; // rsi
  int v27; // [rsp+8h] [rbp-188h]
  __int64 v28; // [rsp+18h] [rbp-178h]
  __int64 *v29; // [rsp+20h] [rbp-170h] BYREF
  __int64 v30; // [rsp+28h] [rbp-168h] BYREF
  unsigned __int64 v31[2]; // [rsp+30h] [rbp-160h] BYREF
  __int16 v32; // [rsp+40h] [rbp-150h]
  const char *v33; // [rsp+50h] [rbp-140h] BYREF
  __int64 v34; // [rsp+58h] [rbp-138h]
  _BYTE v35[304]; // [rsp+60h] [rbp-130h] BYREF

  v8 = *(_DWORD *)(a1 + 64);
  if ( v8 != 52 && v8 != 375 && v8 != 369 )
  {
    v16 = *(_QWORD *)(a1 + 56);
    v35[1] = 1;
    v33 = "expected scope value for catchswitch";
    v35[0] = 3;
    return (unsigned int)sub_38814C0(a1 + 8, v16, (__int64)&v33);
  }
  v9 = sub_16432D0(*(_QWORD **)a1);
  if ( (unsigned __int8)sub_38A1070((__int64 **)a1, v9, &v29, a3, a4, a5, a6)
    || (unsigned __int8)sub_388AF10(a1, 6, "expected '[' with catchswitch labels") )
  {
    return 1;
  }
  v33 = v35;
  v34 = 0x2000000000LL;
  while ( 1 )
  {
    v31[0] = 0;
    v10 = sub_38AB2F0(a1, &v30, v31, a3, a4, a5, a6);
    if ( (_BYTE)v10 )
      goto LABEL_13;
    v14 = (unsigned int)v34;
    if ( (unsigned int)v34 >= HIDWORD(v34) )
    {
      sub_16CD150((__int64)&v33, v35, 0, 8, v12, v13);
      v14 = (unsigned int)v34;
    }
    *(_QWORD *)&v33[8 * v14] = v30;
    LODWORD(v34) = v34 + 1;
    if ( *(_DWORD *)(a1 + 64) != 4 )
      break;
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  }
  v10 = sub_388AF10(a1, 7, "expected ']' after catchswitch labels");
  if ( (_BYTE)v10 || (v10 = sub_388AF10(a1, 63, "expected 'unwind' after catchswitch scope"), (_BYTE)v10) )
  {
LABEL_13:
    v15 = v33;
    goto LABEL_14;
  }
  v30 = 0;
  if ( *(_DWORD *)(a1 + 64) == 53 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    v17 = sub_388AF10(a1, 54, "expected 'caller' in catchswitch");
    if ( !(_BYTE)v17 )
      goto LABEL_20;
  }
  else
  {
    v31[0] = 0;
    v17 = sub_38AB2F0(a1, &v30, v31, a3, a4, a5, a6);
    if ( !(_BYTE)v17 )
    {
LABEL_20:
      v32 = 257;
      v18 = v29;
      v27 = v34;
      v28 = v30;
      v19 = sub_1648B60(64);
      v24 = v19;
      if ( v19 )
        sub_15F7B50(v19, v18, v28, v27, (__int64)v31, 0);
      v25 = (__int64 *)v33;
      v15 = &v33[8 * (unsigned int)v34];
      if ( v33 != v15 )
      {
        do
        {
          v26 = *v25++;
          sub_15F7DB0(v24, v26, v20, v21, v22, v23);
        }
        while ( v15 != (const char *)v25 );
        v15 = v33;
      }
      *a2 = v24;
      goto LABEL_14;
    }
  }
  v15 = v33;
  v10 = v17;
LABEL_14:
  if ( v15 != v35 )
    _libc_free((unsigned __int64)v15);
  return v10;
}
