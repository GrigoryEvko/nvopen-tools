// Function: sub_D10A50
// Address: 0xd10a50
//
__int64 __fastcall sub_D10A50(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // r13
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v10[2]; // [rsp-68h] [rbp-68h] BYREF
  __int64 v11; // [rsp-58h] [rbp-58h]
  char v12; // [rsp-50h] [rbp-50h]
  __int64 v13; // [rsp-48h] [rbp-48h]

  v1 = *(__int64 **)(a1 + 64);
  if ( *(_QWORD *)(a1 + 72) - (_QWORD)v1 > 8u )
    return 1;
  v2 = *v1;
  v3 = *(_QWORD *)(*v1 + 24);
  v4 = *(_QWORD *)(*v1 + 16);
  if ( v3 != v4 )
  {
    while ( 1 )
    {
      v12 = 0;
      if ( !*(_BYTE *)(v4 + 24) )
        break;
      v10[0] = 6;
      v10[1] = 0;
      v6 = *(_QWORD *)(v4 + 16);
      v11 = v6;
      if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
        sub_BD6050(v10, *(_QWORD *)v4 & 0xFFFFFFFFFFFFFFF8LL);
      v12 = 1;
      v13 = *(_QWORD *)(v4 + 32);
      v7 = sub_D0EE50((__int64)v10);
      v12 = 0;
      v5 = v7;
      if ( v11 == -4096 || v11 == 0 || v11 == -8192 )
        goto LABEL_5;
      sub_BD60C0(v10);
      if ( v2 == v5 )
        return 1;
LABEL_6:
      v4 += 40;
      if ( v3 == v4 )
        return 0;
    }
    v13 = *(_QWORD *)(v4 + 32);
    v5 = sub_D0EE50((__int64)v10);
LABEL_5:
    if ( v2 == v5 )
      return 1;
    goto LABEL_6;
  }
  return 0;
}
