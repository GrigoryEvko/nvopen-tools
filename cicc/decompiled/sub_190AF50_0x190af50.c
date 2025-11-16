// Function: sub_190AF50
// Address: 0x190af50
//
__int64 ***__fastcall sub_190AF50(__int64 ***a1, __int64 a2, __int64 *a3)
{
  const char *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 *v6; // r15
  unsigned __int64 v7; // rax
  __int64 ***v8; // rax
  __int64 v9; // r14
  __int64 v10; // r12
  __int64 *v12; // rbx
  unsigned __int64 v13; // rax
  _BYTE v15[64]; // [rsp+10h] [rbp-C0h] BYREF
  unsigned __int64 v16[2]; // [rsp+50h] [rbp-80h] BYREF
  _BYTE v17[112]; // [rsp+60h] [rbp-70h] BYREF

  if ( *(_DWORD *)(a2 + 8) == 1 && sub_15CC890(a3[3], **(_QWORD **)a2, (__int64)a1[5]) )
  {
    v12 = *(__int64 **)a2;
    v13 = sub_157EBA0(**(_QWORD **)a2);
    return sub_190AE30((__int64)(v12 + 1), a1, v13, a3);
  }
  v16[0] = (unsigned __int64)v17;
  v16[1] = 0x800000000LL;
  sub_1B3B830(v15, v16);
  v3 = sub_1649960((__int64)a1);
  sub_1B3B8C0(v15, *a1, v3, v4);
  v5 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v5 )
  {
    v6 = *(__int64 **)a2;
    while ( 1 )
    {
      v9 = *v6;
      if ( (unsigned __int8)sub_1B3BC80(v15, *v6) )
        goto LABEL_5;
      if ( (__int64 **)v9 == a1[5]
        && (((v6[1] >> 1) & 3) == 0 || ((v6[1] >> 1) & 3) == 1)
        && a1 == (__int64 ***)(v6[1] & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v6 += 3;
        if ( (__int64 *)v5 == v6 )
          break;
      }
      else
      {
        v7 = sub_157EBA0(*v6);
        v8 = sub_190AE30((__int64)(v6 + 1), a1, v7, a3);
        sub_1B3BE00(v15, v9, v8);
LABEL_5:
        v6 += 3;
        if ( (__int64 *)v5 == v6 )
          break;
      }
    }
  }
  v10 = sub_1B40B40(v15);
  sub_1B3B860(v15);
  if ( (_BYTE *)v16[0] != v17 )
    _libc_free(v16[0]);
  return (__int64 ***)v10;
}
