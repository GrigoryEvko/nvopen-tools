// Function: sub_1DF76E0
// Address: 0x1df76e0
//
__int64 __fastcall sub_1DF76E0(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r14d
  __int64 v6; // r13
  char *v8; // rax
  char *v9; // r11
  char v10; // cl
  char *v11; // rax
  char v12; // r12
  char *v13; // r9
  int v14; // esi
  __int64 v15; // rdi
  __int64 v16; // r10
  __int64 v17; // rdi
  unsigned int v18; // r8d
  _WORD *v19; // r15
  unsigned __int16 v20; // cx
  __int16 *v21; // r8
  unsigned int v22; // r15d
  _WORD *v23; // rdi
  int v24; // esi
  unsigned __int16 *v25; // r10
  unsigned int i; // edi
  bool v27; // cf
  __int16 *v28; // r15
  __int16 v29; // r8
  int v30; // ecx
  int v31; // edi
  __int64 v32; // [rsp+0h] [rbp-48h]
  int v34; // [rsp+14h] [rbp-34h]

  v34 = *(_DWORD *)(a1 + 296);
  if ( v34 )
  {
    v5 = a2 & 0x1F;
    v6 = 4LL * ((unsigned int)a2 >> 5);
    v32 = 24LL * (unsigned int)a2;
    do
    {
      if ( a3 == a4 )
        break;
      while ( (unsigned __int16)(**(_WORD **)(a3 + 16) - 12) <= 1u )
      {
        if ( (*(_BYTE *)a3 & 4) == 0 )
        {
          while ( (*(_BYTE *)(a3 + 46) & 8) != 0 )
            a3 = *(_QWORD *)(a3 + 8);
        }
        a3 = *(_QWORD *)(a3 + 8);
        if ( a3 == a4 )
          return 0;
      }
      if ( a4 == a3 )
        break;
      v8 = *(char **)(a3 + 32);
      v9 = &v8[40 * *(unsigned int *)(a3 + 40)];
      if ( v9 != v8 )
      {
        v10 = *v8;
        v11 = v8 + 40;
        v12 = 0;
        v13 = v11 - 40;
        if ( v10 == 12 )
        {
          while ( 1 )
          {
            v30 = *(_DWORD *)(*((_QWORD *)v11 - 2) + v6);
            if ( !_bittest(&v30, v5) )
              break;
LABEL_29:
            if ( v9 == v11 )
            {
              if ( v12 )
                return 1;
              goto LABEL_31;
            }
LABEL_27:
            v10 = *v11;
            v11 += 40;
            v13 = v11 - 40;
            if ( v10 != 12 )
              goto LABEL_14;
          }
        }
        else
        {
LABEL_14:
          if ( v10 )
            goto LABEL_29;
          v14 = *((_DWORD *)v11 - 8);
          if ( !v14 )
            goto LABEL_29;
          if ( a2 != v14 )
          {
            if ( v14 < 0 || a2 < 0 )
              goto LABEL_29;
            v15 = *(_QWORD *)(a1 + 240);
            v16 = *(_QWORD *)(v15 + 8);
            v17 = *(_QWORD *)(v15 + 56);
            v18 = *(_DWORD *)(v16 + 24LL * (unsigned int)v14 + 16);
            v19 = (_WORD *)(v17 + 2LL * (v18 >> 4));
            v20 = *v19 + v14 * (v18 & 0xF);
            v21 = v19 + 1;
            v22 = v20;
            LODWORD(v16) = *(_DWORD *)(v16 + v32 + 16);
            v24 = a2 * (v16 & 0xF);
            v23 = (_WORD *)(v17 + 2LL * ((unsigned int)v16 >> 4));
            LOWORD(v24) = *v23 + a2 * (v16 & 0xF);
            v25 = v23 + 1;
            for ( i = (unsigned __int16)v24; ; i = (unsigned __int16)v24 )
            {
              v27 = v22 < i;
              if ( v22 == i )
                break;
              while ( v27 )
              {
                v28 = v21 + 1;
                v29 = *v21;
                v20 += v29;
                if ( !v29 )
                  goto LABEL_29;
                v21 = v28;
                v22 = v20;
                v27 = v20 < i;
                if ( v20 == i )
                  goto LABEL_24;
              }
              v31 = *v25;
              if ( !(_WORD)v31 )
                goto LABEL_29;
              v24 += v31;
              ++v25;
            }
          }
LABEL_24:
          if ( (v13[3] & 0x10) == 0 )
            return 0;
        }
        if ( v9 != v11 )
        {
          v12 = 1;
          goto LABEL_27;
        }
        return 1;
      }
LABEL_31:
      --v34;
      if ( (*(_BYTE *)a3 & 4) == 0 )
      {
        while ( (*(_BYTE *)(a3 + 46) & 8) != 0 )
          a3 = *(_QWORD *)(a3 + 8);
      }
      a3 = *(_QWORD *)(a3 + 8);
    }
    while ( v34 );
  }
  return 0;
}
