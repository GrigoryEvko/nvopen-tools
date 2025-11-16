// Function: sub_1CC55E0
// Address: 0x1cc55e0
//
__int64 __fastcall sub_1CC55E0(unsigned __int8 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // r12
  unsigned __int8 v5; // r11
  __int64 v6; // rdx
  unsigned int v7; // r12d
  unsigned int v8; // eax
  unsigned __int8 v9; // r11
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rdx
  unsigned int v14; // r12d
  unsigned int v15; // eax
  __int64 v16; // [rsp+0h] [rbp-50h]
  unsigned __int8 v17; // [rsp+8h] [rbp-48h]
  unsigned __int8 v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]
  unsigned __int8 v21; // [rsp+1Eh] [rbp-32h]
  unsigned __int8 v22; // [rsp+1Fh] [rbp-31h]

  if ( *a1 || a1[1] )
  {
    v2 = sub_1632FA0(a2);
    v3 = *(_QWORD *)(a2 + 16);
    v20 = v2;
    if ( v3 != a2 + 8 )
    {
      v22 = 0;
      while ( 1 )
      {
        if ( !v3 )
          BUG();
        if ( (*(_BYTE *)(v3 - 24) & 0xFu) - 4 <= 1 )
          goto LABEL_5;
        v4 = *(_QWORD *)(v3 - 56);
        if ( sub_15E4F60(v3 - 56) )
          goto LABEL_5;
        v5 = *a1;
        if ( *a1 && *(_DWORD *)(v4 + 8) >> 8 == 3 )
        {
          v11 = *(_QWORD *)(v3 - 32);
          if ( *(_BYTE *)(v11 + 8) != 14 || *(_DWORD *)(*(_QWORD *)(v3 - 56) + 8LL) >> 8 != 3 )
            goto LABEL_5;
          v12 = *(_QWORD *)(v11 + 24);
          v13 = *(_QWORD *)(v11 + 32);
          v14 = (unsigned int)(1 << (*(_DWORD *)(v3 - 24) >> 15)) >> 1;
          if ( v14 )
          {
            if ( !v13 )
              goto LABEL_23;
          }
          else
          {
            v16 = *(_QWORD *)(v11 + 32);
            v21 = *a1;
            v19 = *(_QWORD *)(v11 + 24);
            v15 = sub_15AAE50(v20, v19);
            LODWORD(v13) = v16;
            v12 = v19;
            v5 = v21;
            v14 = v15;
            if ( !v16 )
            {
              if ( !v15 )
                goto LABEL_5;
LABEL_23:
              v22 = v5;
              goto LABEL_5;
            }
          }
          v18 = v5;
          v8 = sub_1CC52F0(v20, v14, v13, v12);
          v9 = v18;
          if ( v8 == v14 )
            goto LABEL_5;
        }
        else
        {
          if ( !a1[1] )
            goto LABEL_5;
          if ( *(_DWORD *)(v4 + 8) >> 8 != 1 )
            goto LABEL_5;
          v6 = *(_QWORD *)(v3 - 56);
          if ( *(_DWORD *)(v6 + 8) >> 8 != 1 )
            goto LABEL_5;
          if ( *(_BYTE *)(*(_QWORD *)(v6 + 24) + 8LL) != 14 )
            goto LABEL_5;
          v17 = a1[1];
          v7 = (unsigned int)(1 << (*(_DWORD *)(v3 - 24) >> 15)) >> 1;
          v8 = sub_1CC52F0(v20, v7, *(_QWORD *)(*(_QWORD *)(v3 - 32) + 32LL), *(_QWORD *)(*(_QWORD *)(v3 - 32) + 24LL));
          v9 = v17;
          if ( v7 == v8 )
            goto LABEL_5;
        }
        v22 = v9;
        sub_15E4CC0(v3 - 56, v8);
LABEL_5:
        v3 = *(_QWORD *)(v3 + 8);
        if ( a2 + 8 == v3 )
          return v22;
      }
    }
  }
  return 0;
}
