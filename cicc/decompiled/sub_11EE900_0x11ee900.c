// Function: sub_11EE900
// Address: 0x11ee900
//
unsigned __int64 __fastcall sub_11EE900(__int64 a1, __int64 a2, unsigned int **a3)
{
  __int64 *v4; // rax
  __int64 v5; // r12
  unsigned __int64 v6; // r15
  __int64 v7; // r12
  __int64 v9; // r13
  __int64 v10; // rax
  _BYTE *v11; // r10
  _BYTE *v12; // r9
  const char *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // r12
  __int64 v21; // rax
  _BYTE *v22; // [rsp+0h] [rbp-80h]
  __int64 v23; // [rsp+8h] [rbp-78h]
  _BYTE *v24; // [rsp+8h] [rbp-78h]
  __int64 v25; // [rsp+8h] [rbp-78h]
  _BYTE *v26; // [rsp+8h] [rbp-78h]
  _BYTE *v27; // [rsp+8h] [rbp-78h]
  __int64 v28; // [rsp+18h] [rbp-68h]
  _QWORD v29[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v30; // [rsp+40h] [rbp-40h]

  v4 = (__int64 *)sub_B43CA0(a2);
  v5 = *(_QWORD *)(a2 - 32);
  if ( v5 )
  {
    if ( *(_BYTE *)v5 )
    {
      v5 = 0;
    }
    else if ( *(_QWORD *)(a2 + 80) != *(_QWORD *)(v5 + 24) )
    {
      v5 = 0;
    }
  }
  if ( sub_11C99B0(v4, *(__int64 **)(a1 + 24), 0x1C1u)
    && ((v13 = sub_BD5D20(v5), v14 == 4) && *(_DWORD *)v13 == 1953657203 || *(_DWORD *)(v5 + 36) == 335) )
  {
    v6 = sub_11DB650(a2, (__int64)a3, 0, *(__int64 **)(a1 + 24), 1);
  }
  else
  {
    v6 = 0;
  }
  v7 = sub_11EE290(a1, a2, (__int64)a3);
  if ( !v7 )
  {
    v7 = v6;
    if ( sub_B45190(a2) )
    {
      v9 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      if ( *(_BYTE *)v9 == 47 )
      {
        if ( sub_B45190(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) )
        {
          if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
            v10 = *(_QWORD *)(v9 - 8);
          else
            v10 = v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
          v11 = *(_BYTE **)v10;
          v12 = *(_BYTE **)(v10 + 32);
          if ( v12 == *(_BYTE **)v10 )
          {
            v7 = v6;
            if ( !v12 )
              return v7;
            v23 = *(_QWORD *)(v10 + 32);
            v29[0] = "fabs";
            v30 = 259;
            BYTE4(v28) = 1;
            LODWORD(v28) = sub_B45210(v9);
            v7 = sub_B33BC0((__int64)a3, 0xAAu, v23, v28, (__int64)v29);
            goto LABEL_27;
          }
          if ( *v11 == 47
            && (v15 = *((_QWORD *)v11 - 8)) != 0
            && (v16 = *((_QWORD *)v11 - 4), v15 == v16)
            && v16
            && (v22 = v12, v24 = v11, v17 = sub_B45190((__int64)v11), v11 = v24, v12 = v22, v17) )
          {
            v29[0] = "fabs";
            v30 = 259;
            BYTE4(v28) = 1;
            LODWORD(v28) = sub_B45210(v9);
            v18 = sub_B33BC0((__int64)a3, 0xAAu, v15, v28, (__int64)v29);
            v19 = (__int64)v22;
            v7 = v18;
            if ( !v22 )
            {
LABEL_27:
              if ( v7 && *(_BYTE *)v7 == 85 )
                *(_WORD *)(v7 + 2) = *(_WORD *)(v7 + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
              return v7;
            }
          }
          else
          {
            if ( *v12 != 47 )
              return v6;
            v20 = *((_QWORD *)v12 - 8);
            v27 = v11;
            if ( !v20 )
              return v6;
            v21 = *((_QWORD *)v12 - 4);
            if ( v20 != v21 || !v21 || !sub_B45190((__int64)v12) )
              return v6;
            v29[0] = "fabs";
            v30 = 259;
            BYTE4(v28) = 1;
            LODWORD(v28) = sub_B45210(v9);
            v7 = sub_B33BC0((__int64)a3, 0xAAu, v20, v28, (__int64)v29);
            v19 = (__int64)v27;
          }
          v25 = v19;
          v29[0] = "sqrt";
          v30 = 259;
          BYTE4(v28) = 1;
          LODWORD(v28) = sub_B45210(v9);
          v26 = (_BYTE *)sub_B33BC0((__int64)a3, 0x14Fu, v25, v28, (__int64)v29);
          v30 = 257;
          BYTE4(v28) = 1;
          LODWORD(v28) = sub_B45210(v9);
          v7 = sub_A826E0(a3, (_BYTE *)v7, v26, v28, (__int64)v29, 0);
          goto LABEL_27;
        }
      }
    }
  }
  return v7;
}
