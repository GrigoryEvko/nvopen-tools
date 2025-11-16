// Function: sub_3577FF0
// Address: 0x3577ff0
//
void __fastcall sub_3577FF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rax
  __int64 *v8; // rdx
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 *v12; // rax
  char v13; // al
  _BYTE *v14; // rsi
  char v15; // dl
  __int64 v16[3]; // [rsp+8h] [rbp-18h] BYREF

  if ( *(_BYTE *)(a1 + 1284) )
  {
    v7 = *(__int64 **)(a1 + 1264);
    v8 = &v7[*(unsigned int *)(a1 + 1276)];
    if ( v7 != v8 )
    {
      while ( a2 != *v7 )
      {
        if ( v8 == ++v7 )
          goto LABEL_8;
      }
      return;
    }
  }
  else if ( sub_C8CA60(a1 + 1256, a2) )
  {
    return;
  }
LABEL_8:
  v9 = *(_DWORD *)(a2 + 44);
  if ( (v9 & 4) == 0 && (v9 & 8) != 0 )
    LOBYTE(v10) = sub_2E88A90(a2, 512, 1);
  else
    v10 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 9) & 1LL;
  if ( !(_BYTE)v10 )
  {
    v13 = sub_3577E50(a1, a2);
    goto LABEL_19;
  }
  v11 = *(_QWORD *)(a2 + 24);
  if ( !*(_BYTE *)(a1 + 300) )
  {
LABEL_26:
    sub_C8CC70(a1 + 272, v11, (__int64)v8, a4, a5, a6);
    v13 = v15;
LABEL_19:
    if ( !v13 )
      return;
    goto LABEL_20;
  }
  v12 = *(__int64 **)(a1 + 280);
  a4 = *(unsigned int *)(a1 + 292);
  v8 = &v12[a4];
  if ( v12 == v8 )
  {
LABEL_25:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 288) )
    {
      *(_DWORD *)(a1 + 292) = a4 + 1;
      *v8 = v11;
      ++*(_QWORD *)(a1 + 272);
LABEL_20:
      v16[0] = a2;
      v14 = *(_BYTE **)(a1 + 568);
      if ( v14 == *(_BYTE **)(a1 + 576) )
      {
        sub_35767D0(a1 + 560, v14, v16);
      }
      else
      {
        if ( v14 )
        {
          *(_QWORD *)v14 = a2;
          v14 = *(_BYTE **)(a1 + 568);
        }
        *(_QWORD *)(a1 + 568) = v14 + 8;
      }
      return;
    }
    goto LABEL_26;
  }
  while ( v11 != *v12 )
  {
    if ( v8 == ++v12 )
      goto LABEL_25;
  }
}
