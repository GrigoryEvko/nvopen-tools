// Function: sub_C42E80
// Address: 0xc42e80
//
__int64 __fastcall sub_C42E80(__int64 a1, _BYTE *a2, unsigned __int64 a3, unsigned __int64 a4, char a5)
{
  __int64 v5; // r12
  unsigned __int64 v6; // rbx
  const char *v7; // rax
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // r13
  unsigned int v11; // r14d
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v15; // r15
  unsigned __int8 *v16; // r14
  int v18; // eax
  char v19; // al
  unsigned __int8 *v20; // r9
  char v21; // al
  unsigned __int64 v22; // [rsp+8h] [rbp-68h]
  const char *v23; // [rsp+10h] [rbp-60h] BYREF
  char v24; // [rsp+30h] [rbp-40h]
  char v25; // [rsp+31h] [rbp-3Fh]

  v5 = a1;
  v6 = a4;
  if ( a4 )
  {
    v15 = (__int64)a2;
    v16 = (unsigned __int8 *)a3;
    a2 = (_BYTE *)a3;
    a1 = v15;
    v22 = a4;
    v18 = sub_C36F70((_DWORD **)v15, (_BYTE *)a3, a4);
    a4 = v22;
    if ( (_BYTE)v18 )
    {
      v19 = *(_BYTE *)(v5 + 8);
      *(_DWORD *)v5 = 0;
      *(_BYTE *)(v5 + 8) = v19 & 0xFC | 2;
      return v5;
    }
    v20 = v16;
    LOBYTE(v18) = *v16 == 45;
    a3 = (unsigned int)(8 * v18);
    v21 = (8 * v18) | *(_BYTE *)(v15 + 20) & 0xF7;
    *(_BYTE *)(v15 + 20) = a3 | *(_BYTE *)(v15 + 20) & 0xF7;
    if ( (v21 & 8) != 0 && !*(_BYTE *)(*(_QWORD *)v15 + 25LL) )
      BUG();
    if ( ((*v16 - 43) & 0xFD) == 0 && (v20 = v16 + 1, a4 = v6 - 1, v6 == 1) )
    {
      v25 = 1;
      v7 = "String has no digits";
    }
    else
    {
      if ( a4 == 1 || *v20 != 48 || (v20[1] & 0xDF) != 0x58 )
      {
        sub_C42160(v5, (__int64 *)v15, v20, a4, a5);
        return v5;
      }
      if ( a4 != 2 )
      {
        sub_C42970(v5, v15, v20 + 2, a4 - 2, a5);
        return v5;
      }
      v25 = 1;
      v7 = "Invalid string";
    }
  }
  else
  {
    v25 = 1;
    v7 = "Invalid string length";
  }
  v23 = v7;
  v24 = 3;
  v8 = sub_C63BB0(a1, a2, a3, a4);
  v10 = v9;
  v11 = v8;
  v12 = sub_22077B0(64);
  v13 = v12;
  if ( v12 )
    sub_C63EB0(v12, &v23, v11, v10);
  *(_BYTE *)(v5 + 8) |= 3u;
  *(_QWORD *)v5 = v13 & 0xFFFFFFFFFFFFFFFELL;
  return v5;
}
