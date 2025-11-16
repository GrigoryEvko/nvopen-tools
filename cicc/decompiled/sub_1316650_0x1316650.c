// Function: sub_1316650
// Address: 0x1316650
//
_QWORD *__fastcall sub_1316650(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v8; // rax
  unsigned __int8 v9; // cl
  _QWORD *v10; // r12
  __int64 (__fastcall **v12)(int, int, int, int, int, int, int); // rax
  __int64 v13; // rax
  _BYTE v14[49]; // [rsp+Fh] [rbp-31h] BYREF

  v5 = a4;
  v14[0] = 0;
  v8 = sub_1316370(a2);
  if ( !qword_4F969E0 )
    goto LABEL_2;
  v12 = *(__int64 (__fastcall ***)(int, int, int, int, int, int, int))(v8 + 8);
  if ( !a1 || v12 != &off_49E8020 )
    goto LABEL_2;
  v13 = *(_QWORD *)(a1 + 120);
  if ( v13 != 1 )
  {
    *(_QWORD *)(a1 + 120) = v13 - 1;
LABEL_2:
    v9 = 0;
    goto LABEL_3;
  }
  v9 = 1;
  *(_QWORD *)(a1 + 120) = qword_4F969E0;
LABEL_3:
  v10 = (_QWORD *)sub_130B510(a1, a2 + 10648, *(_QWORD *)(a5 + 8), 4096, 1u, a3, 0, v9, (__int64)v14);
  if ( v14[0] )
    sub_1314D40(a1, a2);
  if ( v10 )
  {
    *v10 = *v10 & 0xFFFFF0000FFFFFFFLL | (v5 << 38) | ((unsigned __int64)*(unsigned int *)(a5 + 16) << 28);
    sub_131C8A0(v10 + 8);
  }
  return v10;
}
