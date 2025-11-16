// Function: sub_388CC90
// Address: 0x388cc90
//
__int64 __fastcall sub_388CC90(__int64 a1, _DWORD *a2, __int64 a3)
{
  __int64 v3; // r15
  int v6; // eax
  unsigned __int64 v7; // rsi
  const char *v8; // rax
  unsigned int v9; // eax
  unsigned int v10; // r12d
  int v12; // eax
  bool v13; // zf
  unsigned __int64 v14; // [rsp+8h] [rbp-68h]
  int v15; // [rsp+1Ch] [rbp-54h] BYREF
  _QWORD v16[2]; // [rsp+20h] [rbp-50h] BYREF
  char v17; // [rsp+30h] [rbp-40h]
  char v18; // [rsp+31h] [rbp-3Fh]

  v3 = a1 + 8;
  v6 = sub_3887100(a1 + 8);
  v7 = *(_QWORD *)(a1 + 56);
  *(_DWORD *)(a1 + 64) = v6;
  if ( v6 == 12 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(v3);
    v10 = sub_388BA90(a1, a2);
    if ( (_BYTE)v10 )
      return v10;
    v12 = *(_DWORD *)(a1 + 64);
    if ( v12 == 4 )
    {
      *(_DWORD *)(a1 + 64) = sub_3887100(v3);
      v14 = *(_QWORD *)(a1 + 56);
      v9 = sub_388BA90(a1, &v15);
      if ( (_BYTE)v9 )
        return v9;
      if ( *a2 == v15 )
      {
        v18 = 1;
        v17 = 3;
        v16[0] = "'allocsize' indices can't refer to the same parameter";
        return (unsigned int)sub_38814C0(v3, v14, (__int64)v16);
      }
      v13 = *(_BYTE *)(a3 + 4) == 0;
      *(_DWORD *)a3 = v15;
      if ( v13 )
        *(_BYTE *)(a3 + 4) = 1;
    }
    else
    {
      if ( !*(_BYTE *)(a3 + 4) )
      {
LABEL_11:
        v7 = *(_QWORD *)(a1 + 56);
        if ( v12 == 13 )
        {
          *(_DWORD *)(a1 + 64) = sub_3887100(v3);
          return v10;
        }
        v18 = 1;
        v8 = "expected ')'";
        goto LABEL_3;
      }
      *(_BYTE *)(a3 + 4) = 0;
    }
    v12 = *(_DWORD *)(a1 + 64);
    goto LABEL_11;
  }
  v18 = 1;
  v8 = "expected '('";
LABEL_3:
  v16[0] = v8;
  v17 = 3;
  return (unsigned int)sub_38814C0(v3, v7, (__int64)v16);
}
