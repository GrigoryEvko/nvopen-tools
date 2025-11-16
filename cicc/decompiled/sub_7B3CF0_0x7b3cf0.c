// Function: sub_7B3CF0
// Address: 0x7b3cf0
//
__int64 __fastcall sub_7B3CF0(unsigned __int8 *a1, int *a2, int a3)
{
  unsigned __int64 v5; // rdi
  int v6; // r12d
  __int64 result; // rax
  unsigned __int64 v8; // rax
  int v9; // eax
  unsigned int v10; // [rsp+4h] [rbp-2Ch] BYREF
  unsigned __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = *a1;
  if ( (_BYTE)v5 == 92 )
  {
    v5 = 92;
    v6 = 1;
    if ( (a1[1] & 0xDF) == 0x55 && unk_4D042A0 )
    {
      v11[0] = (unsigned __int64)a1;
      v8 = sub_7B39D0(v11, 0, 0, 1);
      v9 = sub_7AC070(v8, a3);
      v6 = LODWORD(v11[0]) - (_DWORD)a1;
      result = v9 == 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v11[0] = v5;
  v6 = 1;
  if ( v5 <= 0x7F || !unk_4F064A8 )
  {
LABEL_4:
    result = dword_4F059C0[v5];
    if ( (_DWORD)result )
    {
      result = 1;
      if ( a3 )
        result = (unsigned int)(v5 - 48) > 9;
    }
    goto LABEL_7;
  }
  v6 = sub_722680(a1, v11, (int *)&v10, 0);
  result = v10;
  if ( v10 )
  {
    v5 = 0;
    goto LABEL_4;
  }
  v5 = v11[0];
  if ( v11[0] <= 0xFF )
    goto LABEL_4;
  if ( v11[0] - 55296 > 0x7FF )
    result = (unsigned int)sub_7AC070(v11[0], a3) == 0;
LABEL_7:
  if ( a2 )
    *a2 = v6;
  return result;
}
