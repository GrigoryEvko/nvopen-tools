// Function: sub_39365E0
// Address: 0x39365e0
//
__int64 __fastcall sub_39365E0(__int64 *a1, unsigned __int64 a2, int a3, unsigned int a4)
{
  unsigned __int64 v7; // rax
  unsigned int v8; // ebx
  char v9; // cl
  int v10; // edx
  unsigned __int64 v11; // r10
  unsigned __int64 v12; // rax
  int v13; // r15d
  char v14; // cl
  const char *v15; // rcx
  int v17; // [rsp+Ch] [rbp-34h]
  int v18; // [rsp+Ch] [rbp-34h]

  while ( a2 )
  {
    v7 = a2;
    v8 = 0;
    if ( (a2 & 1) != 0 )
    {
      v10 = a3;
      v11 = a2 >> 1;
      v12 = ~(a2 >> 1);
    }
    else
    {
      do
      {
        v7 >>= 1;
        v9 = v8++;
      }
      while ( (v7 & 1) == 0 );
      v10 = a3 + v8;
      v11 = a2 >> (v9 + 2);
      v12 = ~v11;
      if ( v11 == -1 )
      {
        v11 = -1;
LABEL_17:
        a2 = v11 >> 1;
        if ( (_BYTE)a4 )
        {
          v18 = v10;
          sub_1688520(a1, 44);
          v10 = v18;
        }
        goto LABEL_19;
      }
    }
    if ( (v12 & 1) != 0 )
      goto LABEL_17;
    v13 = 0;
    do
    {
      v12 >>= 1;
      v14 = v13++;
    }
    while ( (v12 & 1) == 0 );
    v8 += v13;
    a2 = v11 >> (v14 + 2);
    if ( (_BYTE)a4 )
    {
      v17 = v10;
      sub_1688520(a1, 44);
      v10 = v17;
    }
    if ( v13 )
    {
      v15 = "-";
      if ( v13 == 1 )
        v15 = ",";
      sub_1688630(a1, "%d%s%d", v10, v15, a3 + v8);
      goto LABEL_14;
    }
LABEL_19:
    sub_1688630(a1, "%d", v10);
LABEL_14:
    a3 += v8 + 2;
    a4 = 1;
    if ( v8 > 0x3E )
      return a4;
  }
  return a4;
}
