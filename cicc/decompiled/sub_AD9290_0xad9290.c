// Function: sub_AD9290
// Address: 0xad9290
//
unsigned __int8 *__fastcall sub_AD9290(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // r12
  unsigned __int8 *v9; // r13
  int v10; // edx
  _QWORD *i; // rbx
  __int64 v13; // [rsp+8h] [rbp-58h]
  __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  _QWORD *v15; // [rsp+18h] [rbp-48h]

  v2 = a1;
  if ( (unsigned int)*(unsigned __int8 *)(a1 + 8) - 17 <= 1 )
    a1 = **(_QWORD **)(a1 + 16);
  v3 = sub_BCAC60(a1);
  v7 = sub_C33340(a1, a2, v4, v5, v6);
  v8 = v7;
  if ( v3 == v7 )
  {
    sub_C3C500(&v14, v7, 0);
    a2 = (unsigned __int8)a2;
    if ( v14 != v8 )
      goto LABEL_5;
  }
  else
  {
    sub_C373C0(&v14, v3, 0);
    a2 = (unsigned __int8)a2;
    if ( v14 != v8 )
    {
LABEL_5:
      sub_C37310(&v14, a2);
      goto LABEL_6;
    }
  }
  sub_C3CEB0(&v14, a2);
LABEL_6:
  v9 = (unsigned __int8 *)sub_AC8EA0(*(__int64 **)v2, &v14);
  v10 = *(unsigned __int8 *)(v2 + 8);
  if ( (unsigned int)(v10 - 17) <= 1 )
  {
    BYTE4(v13) = (_BYTE)v10 == 18;
    LODWORD(v13) = *(_DWORD *)(v2 + 32);
    v9 = (unsigned __int8 *)sub_AD5E10(v13, v9);
  }
  if ( v8 == v14 )
  {
    if ( v15 )
    {
      for ( i = &v15[3 * *(v15 - 1)]; v15 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0(&v14);
  }
  return v9;
}
