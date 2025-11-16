// Function: sub_1518010
// Address: 0x1518010
//
__int64 *__fastcall sub_1518010(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  unsigned int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  unsigned int v12; // esi
  int *v13; // r15
  int v14; // r8d
  int v15; // r9d
  _BYTE *v16; // rax
  const char *v17; // rax
  const char *v21; // [rsp+10h] [rbp-50h] BYREF
  char v22; // [rsp+20h] [rbp-40h]
  char v23; // [rsp+21h] [rbp-3Fh]

  if ( a5 )
  {
    v8 = 0;
    while ( 1 )
    {
      v9 = *(unsigned int *)(a2 + 1000);
      if ( !(_DWORD)v9 )
        break;
      v10 = *(_QWORD *)(a2 + 984);
      v11 = *(_QWORD *)(a4 + 8LL * v8);
      v12 = (v9 - 1) & (37 * v11);
      v13 = (int *)(v10 + 8LL * v12);
      v14 = 1;
      v15 = *v13;
      if ( (_DWORD)v11 != *v13 )
      {
        while ( v15 != -1 )
        {
          v12 = (v9 - 1) & (v14 + v12);
          v13 = (int *)(v10 + 8LL * v12);
          v15 = *v13;
          if ( (unsigned int)*(_QWORD *)(a4 + 8LL * v8) == *v13 )
            goto LABEL_5;
          ++v14;
        }
        break;
      }
LABEL_5:
      if ( v13 == (int *)(v10 + 8 * v9) )
        break;
      v16 = (_BYTE *)sub_1517EB0(a2, *(_QWORD *)(a4 + 8LL * (v8 + 1)));
      if ( !v16 || (unsigned __int8)(*v16 - 4) > 0x1Eu )
      {
        v23 = 1;
        v17 = "Invalid metadata attachment";
        goto LABEL_9;
      }
      v8 += 2;
      sub_16267C0(a3, (unsigned int)v13[1], v16);
      if ( a5 == v8 )
        goto LABEL_16;
    }
    v23 = 1;
    v17 = "Invalid ID";
LABEL_9:
    v21 = v17;
    v22 = 3;
    sub_1514BE0(a1, (__int64)&v21);
  }
  else
  {
LABEL_16:
    *a1 = 1;
  }
  return a1;
}
