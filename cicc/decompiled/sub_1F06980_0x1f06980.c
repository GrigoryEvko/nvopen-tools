// Function: sub_1F06980
// Address: 0x1f06980
//
char __fastcall sub_1F06980(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r14
  int v6; // eax
  char v7; // al
  __int16 v8; // ax
  unsigned __int16 *v9; // rbx
  unsigned __int16 *i; // r13
  __int64 v11; // r10
  unsigned int v12; // edi
  unsigned int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int16 v17; // ax
  __int64 *v19; // [rsp+8h] [rbp-58h]
  __int64 *v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  int v22; // [rsp+28h] [rbp-38h]
  int v23; // [rsp+2Ch] [rbp-34h]

  v2 = *(_QWORD *)(a1 + 920);
  v3 = *(_QWORD *)(a1 + 936);
  if ( v3 == v2 + 24 )
  {
    *(_QWORD *)(a1 + 352) = 0;
    goto LABEL_18;
  }
  *(_QWORD *)(a1 + 352) = v3;
  if ( !v3 )
    goto LABEL_18;
  v4 = *(_QWORD *)(v3 + 32);
  v5 = v4 + 40LL * *(unsigned int *)(v3 + 40);
  while ( v5 != v4 )
  {
    while ( *(_BYTE *)v4 || (*(_BYTE *)(v4 + 3) & 0x10) != 0 )
    {
LABEL_11:
      v4 += 40;
      if ( v5 == v4 )
        goto LABEL_12;
    }
    v6 = *(_DWORD *)(v4 + 8);
    if ( v6 <= 0 )
    {
      if ( v6 )
      {
        v7 = *(_BYTE *)(v4 + 4);
        if ( (v7 & 1) == 0 && (v7 & 2) == 0 )
          sub_1F06800(a1, a1 + 344, -858993459 * ((v4 - *(_QWORD *)(v3 + 32)) >> 3));
      }
      goto LABEL_11;
    }
    v4 += 40;
    v23 = v6;
    v21 = a1 + 344;
    v22 = -1;
    sub_1F05660(a1 + 1216, (__int64)&v21);
  }
LABEL_12:
  v8 = *(_WORD *)(v3 + 46);
  if ( (v8 & 4) == 0 && (v8 & 8) != 0 )
    LOBYTE(v2) = sub_1E15D00(v3, 0x10u, 1);
  else
    v2 = (*(_QWORD *)(*(_QWORD *)(v3 + 16) + 8LL) >> 4) & 1LL;
  if ( !(_BYTE)v2 )
  {
    v17 = *(_WORD *)(v3 + 46);
    if ( (v17 & 4) == 0 && (v17 & 8) != 0 )
      LOBYTE(v2) = sub_1E15D00(v3, 0x20u, 1);
    else
      v2 = (*(_QWORD *)(*(_QWORD *)(v3 + 16) + 8LL) >> 5) & 1LL;
    if ( !(_BYTE)v2 )
    {
      v2 = *(_QWORD *)(a1 + 920);
LABEL_18:
      v19 = *(__int64 **)(v2 + 96);
      if ( v19 != *(__int64 **)(v2 + 88) )
      {
        v20 = *(__int64 **)(v2 + 88);
        do
        {
          v9 = *(unsigned __int16 **)(*v20 + 160);
          for ( i = (unsigned __int16 *)sub_1DD77D0(*v20); v9 != i; i += 4 )
          {
            v11 = *i;
            v12 = *(_DWORD *)(a1 + 1224);
            v13 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 1424) + 2 * v11);
            if ( v13 < v12 )
            {
              v14 = *(_QWORD *)(a1 + 1216);
              while ( 1 )
              {
                v15 = v14 + 24LL * v13;
                if ( (_DWORD)v11 == *(_DWORD *)(v15 + 12) )
                {
                  v16 = *(unsigned int *)(v15 + 16);
                  if ( (_DWORD)v16 != -1 && *(_DWORD *)(v14 + 24 * v16 + 20) == -1 )
                    break;
                }
                v13 += 0x10000;
                if ( v12 <= v13 )
                  goto LABEL_31;
              }
              if ( v13 != -1 )
                continue;
            }
LABEL_31:
            v23 = *i;
            v22 = -1;
            v21 = a1 + 344;
            sub_1F05660(a1 + 1216, (__int64)&v21);
          }
          LOBYTE(v2) = (_BYTE)++v20;
        }
        while ( v19 != v20 );
      }
    }
  }
  return v2;
}
