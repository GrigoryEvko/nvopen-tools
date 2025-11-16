// Function: sub_823810
// Address: 0x823810
//
void __fastcall sub_823810(_QWORD *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r12
  void *v8; // rdi
  unsigned __int64 v9; // r13
  __int64 v10; // rax

  if ( a1[1] < a2 )
  {
    v7 = a1[3];
    v8 = (void *)a1[4];
    v9 = v7 * ((v7 + a2 - 1) / v7);
    if ( v8 )
    {
      v10 = realloc(v8);
      if ( v10 )
      {
        v7 = a1[3];
LABEL_5:
        a1[4] = v10;
        a1[3] = 2 * v7;
        a1[1] = v9;
        return;
      }
    }
    else
    {
      v10 = malloc(v7 * ((v7 + a2 - 1) / v7), a2, (v7 + a2 - 1) % v7, a4, a5, a6);
      if ( v10 )
        goto LABEL_5;
    }
    sub_685240(4u);
  }
}
