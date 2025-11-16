// Function: sub_B44330
// Address: 0xb44330
//
void __fastcall sub_B44330(_QWORD *a1, __int64 a2, unsigned __int64 *a3, __int64 a4, char a5)
{
  char v5; // r10
  __int64 v6; // r15
  unsigned __int64 *v9; // r14
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rsi
  __int64 v12; // rax
  char v14; // [rsp+4h] [rbp-3Ch]

  v5 = a5;
  v6 = (__int64)(a1 + 3);
  if ( !*(_BYTE *)(a2 + 40) )
  {
    v9 = (unsigned __int64 *)a1[4];
    if ( (unsigned __int64 *)v6 == a3 || v9 == a3 )
      goto LABEL_10;
    goto LABEL_14;
  }
  if ( !a1[8] || a5 )
  {
LABEL_7:
    v9 = (unsigned __int64 *)a1[4];
    if ( v9 == a3 )
      goto LABEL_17;
    if ( (unsigned __int64 *)v6 == a3 )
    {
      if ( *(_BYTE *)(a2 + 40) != 1 )
        goto LABEL_10;
      goto LABEL_18;
    }
LABEL_14:
    v14 = v5;
    sub_AA4960(a2 + 48, a1[5] + 48LL, v6, 0, (__int64)v9);
    v5 = v14;
    if ( v9 != a3 && v9 != (unsigned __int64 *)v6 )
    {
      v10 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((a1[3] & 0xFFFFFFFFFFFFFFF8LL) + 8) = v9;
      *v9 = *v9 & 7 | a1[3] & 0xFFFFFFFFFFFFFFF8LL;
      v11 = *a3;
      *(_QWORD *)(v10 + 8) = a3;
      v11 &= 0xFFFFFFFFFFFFFFF8LL;
      a1[3] = v11 | a1[3] & 7LL;
      *(_QWORD *)(v11 + 8) = v6;
      *a3 = v10 | *a3 & 7;
    }
LABEL_17:
    if ( *(_BYTE *)(a2 + 40) != 1 )
      goto LABEL_10;
LABEL_18:
    if ( v5 )
      goto LABEL_10;
    goto LABEL_19;
  }
  if ( (unsigned __int64 *)v6 != a3 || (_BYTE)a4 )
  {
    sub_B43CE0((__int64)a1);
    v5 = a5;
    goto LABEL_7;
  }
LABEL_19:
  v12 = sub_AA6190(a1[5], (__int64)a1);
  if ( (_BYTE)a4 != 1 && v12 && v12 + 8 != (*(_QWORD *)(v12 + 8) & 0xFFFFFFFFFFFFFFF8LL) )
    sub_B44050((__int64)a1, a2, (__int64)a3, a4, 0);
LABEL_10:
  if ( (unsigned int)*(unsigned __int8 *)a1 - 30 <= 0xA )
    sub_AA6320(a1[5]);
}
