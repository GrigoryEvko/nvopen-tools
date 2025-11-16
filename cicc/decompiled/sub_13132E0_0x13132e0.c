// Function: sub_13132E0
// Address: 0x13132e0
//
void __fastcall sub_13132E0(_QWORD *a1, char a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // rbx
  unsigned __int64 v5; // r14
  int *v6; // rax
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax

  v2 = unk_4C6F1F0;
  if ( a2 )
  {
    v3 = a1 + 1;
    v4 = a1 + 2;
    v5 = -1;
    a1[1] = a1[103];
    if ( v2 )
    {
      v9 = sub_1310060();
      a1[5] = v9;
      v5 = v9;
      if ( qword_4C6F130[0] < 0LL )
      {
LABEL_4:
        v6 = (int *)sub_134B2D0(a1);
        a1[11] = v6;
        if ( (unsigned __int64)v6 > v5 )
          v6 = (int *)v5;
        goto LABEL_6;
      }
    }
    else if ( qword_4C6F130[0] < 0LL )
    {
      goto LABEL_4;
    }
    v8 = sub_130F8B0(a1);
    a1[9] = v8;
    if ( v5 > v8 )
      v5 = v8;
    goto LABEL_4;
  }
  v3 = a1 + 3;
  v4 = a1 + 4;
  v7 = -1;
  a1[3] = a1[105];
  if ( v2 )
  {
    v10 = sub_1310080();
    a1[6] = v10;
    v7 = v10;
  }
  v6 = (int *)sub_134B340(a1);
  a1[12] = v6;
  if ( (unsigned __int64)v6 > v7 )
    v6 = (int *)v7;
LABEL_6:
  if ( v6 > &dword_400000 )
    v6 = &dword_400000;
  *v4 = (char *)v6 + *v3;
  sub_1313270((__int64)a1);
}
