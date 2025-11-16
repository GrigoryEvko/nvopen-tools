// Function: sub_D23D60
// Address: 0xd23d60
//
__int64 __fastcall sub_D23D60(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  unsigned int v3; // r9d
  __int64 v4; // r8
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r11
  unsigned __int64 *v9; // rsi
  __int64 result; // rax
  int v11; // eax
  int v12; // ebx

  v3 = *(_DWORD *)(a1 + 72);
  v4 = *(_QWORD *)(a1 + 56);
  if ( !v3 )
  {
LABEL_6:
    v7 = (__int64 *)(v4 + 16LL * v3);
    goto LABEL_3;
  }
  v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v4 + 16LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v11 = 1;
    while ( v8 != -4096 )
    {
      v12 = v11 + 1;
      v6 = (v3 - 1) & (v11 + v6);
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v11 = v12;
    }
    goto LABEL_6;
  }
LABEL_3:
  v9 = (unsigned __int64 *)(*(_QWORD *)a1 + 8LL * *((int *)v7 + 2));
  result = 4LL * a3;
  *v9 = result | *v9 & 0xFFFFFFFFFFFFFFFBLL;
  return result;
}
