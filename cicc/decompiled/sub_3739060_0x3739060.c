// Function: sub_3739060
// Address: 0x3739060
//
void __fastcall sub_3739060(__int64 *a1, __int64 a2, char *a3, __int64 a4, __int64 a5, __int64 a6)
{
  char **v6; // r13
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r15
  __int64 v11; // rbx
  _QWORD *v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14[2]; // [rsp+0h] [rbp-60h] BYREF
  _BYTE v15[80]; // [rsp+10h] [rbp-50h] BYREF

  v6 = (char **)a3;
  v8 = a1[26];
  if ( !*(_BYTE *)(v8 + 3688) )
    goto LABEL_9;
  v9 = *((_DWORD *)a3 + 2);
  if ( v9 == 1 )
  {
    if ( (unsigned __int8)sub_3223210(v8) )
    {
      v10 = a1[26];
      v11 = *(_QWORD *)*v6;
      v12 = *(_QWORD **)v11;
      if ( !*(_QWORD *)v11 )
      {
        if ( (*(_BYTE *)(v11 + 9) & 0x70) != 0x20 || *(char *)(v11 + 8) < 0 )
          BUG();
        *(_BYTE *)(v11 + 8) |= 8u;
        v12 = sub_E807D0(*(_QWORD *)(v11 + 24));
        *(_QWORD *)v11 = v12;
      }
      v13 = sub_3222A80(v10, v12[1]);
      a3 = *v6;
      a5 = *(_QWORD *)*v6;
      if ( a5 != v13 )
      {
        v9 = *((_DWORD *)v6 + 2);
        goto LABEL_3;
      }
LABEL_10:
      sub_3738FE0(a1, a2, a5, *(_QWORD *)&a3[16 * *((unsigned int *)v6 + 2) - 8]);
      return;
    }
LABEL_9:
    a3 = *v6;
    a5 = *(_QWORD *)*v6;
    goto LABEL_10;
  }
LABEL_3:
  v14[0] = (unsigned __int64)v15;
  v14[1] = 0x200000000LL;
  if ( v9 )
    sub_37352C0((__int64)v14, v6, (__int64)a3, 0x200000000LL, a5, a6);
  sub_3735F40((__int64)a1, a2, (__int64)v14);
  if ( (_BYTE *)v14[0] != v15 )
    _libc_free(v14[0]);
}
