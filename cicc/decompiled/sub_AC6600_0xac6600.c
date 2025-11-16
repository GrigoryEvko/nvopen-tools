// Function: sub_AC6600
// Address: 0xac6600
//
__int64 __fastcall sub_AC6600(__int64 a1, int *a2, _QWORD *a3)
{
  int v4; // r14d
  unsigned int v6; // eax
  int v7; // ecx
  char v8; // r8
  int v9; // r10d
  int v10; // r14d
  int *v11; // r11
  unsigned int i; // ebx
  int *v13; // rdx
  int v14; // r9d
  unsigned int v15; // ebx
  unsigned int v16; // eax
  char v17; // al
  int v18; // [rsp+8h] [rbp-58h]
  char v19; // [rsp+Fh] [rbp-51h]
  int v20; // [rsp+10h] [rbp-50h]
  int v21; // [rsp+14h] [rbp-4Ch]
  int *v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+28h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v23 = *(_QWORD *)(a1 + 8);
  v6 = sub_C4F140(a2 + 2);
  v7 = *a2;
  v8 = *((_BYTE *)a2 + 4);
  v9 = 1;
  v10 = v4 - 1;
  v11 = 0;
  for ( i = v10
          & (((0xBF58476D1CE4E5B9LL * (v6 | ((unsigned __int64)((unsigned int)(v8 == 0) + 37 * *a2 - 1) << 32))) >> 31)
           ^ (484763065 * v6)); ; i = v10 & v15 )
  {
    v13 = (int *)(v23 + 32LL * i);
    v14 = *v13;
    if ( v7 == *v13 && v8 == *((_BYTE *)v13 + 4) )
    {
      v16 = a2[4];
      if ( v16 == v13[4] )
      {
        if ( v16 <= 0x40 )
        {
          if ( *((_QWORD *)a2 + 1) == *((_QWORD *)v13 + 1) )
          {
LABEL_13:
            *a3 = v13;
            return 1;
          }
        }
        else
        {
          v18 = *v13;
          v19 = v8;
          v20 = v7;
          v21 = v9;
          v22 = v11;
          v17 = sub_C43C50(a2 + 2, v13 + 2);
          v13 = (int *)(v23 + 32LL * i);
          v11 = v22;
          v9 = v21;
          v7 = v20;
          v8 = v19;
          v14 = v18;
          if ( v17 )
            goto LABEL_13;
        }
      }
    }
    if ( v14 == -1 )
      break;
    if ( v14 == -2 && !*((_BYTE *)v13 + 4) && !v13[4] && *((_QWORD *)v13 + 1) == -2 && !v11 )
      v11 = v13;
LABEL_8:
    v15 = v9 + i;
    ++v9;
  }
  if ( !*((_BYTE *)v13 + 4) || v13[4] || *((_QWORD *)v13 + 1) != -1 )
    goto LABEL_8;
  if ( !v11 )
    v11 = v13;
  *a3 = v11;
  return 0;
}
