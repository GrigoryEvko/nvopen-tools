// Function: sub_40EBBB
// Address: 0x40ebbb
//
__int64 __fastcall sub_40EBBB(int a1, int a2, int a3, int a4, const char **a5)
{
  char *v6; // r15
  int v7; // ecx
  int v8; // r8d
  char *v9; // rdx
  int v10; // eax
  int v11; // ecx
  int v12; // r8d
  int v13; // r9d
  _BYTE v15[10]; // [rsp+6h] [rbp-13Ah] BYREF
  _BYTE v16[304]; // [rsp+10h] [rbp-130h] BYREF

  switch ( a4 )
  {
    case 0:
      v6 = "true";
      if ( !*(_BYTE *)a5 )
        v6 = "false";
      goto LABEL_14;
    case 1:
      LODWORD(v6) = *(_DWORD *)a5;
      v7 = a2;
      v8 = a3;
      v9 = "%d";
      goto LABEL_15;
    case 2:
      v6 = (char *)*a5;
      v7 = a2;
      v8 = a3;
      v9 = (char *)"%ld";
      goto LABEL_15;
    case 3:
    case 4:
      LODWORD(v6) = *(_DWORD *)a5;
      v7 = a2;
      v8 = a3;
      v9 = (char *)"%u";
      goto LABEL_15;
    case 5:
      v6 = (char *)*a5;
      v7 = a2;
      v8 = a3;
      v9 = (char *)"%lu";
      goto LABEL_15;
    case 6:
      v6 = (char *)*a5;
      v7 = a2;
      v8 = a3;
      v9 = "%zu";
      goto LABEL_15;
    case 7:
      v6 = (char *)*a5;
      v7 = a2;
      v8 = a3;
      v9 = "%zd";
      goto LABEL_15;
    case 8:
      v6 = v16;
      sub_40E1DF((__int64)v16, 0x100u, "\"%s\"", *a5);
      goto LABEL_14;
    case 9:
      v6 = (char *)*a5;
LABEL_14:
      v8 = a3;
      v7 = a2;
      v9 = "%s";
LABEL_15:
      v10 = sub_40EB59((__int64)v15, 10, (__int64)v9, v7, v8);
      return sub_130F0B0(a1, v10, (_DWORD)v6, v11, v12, v13);
  }
}
