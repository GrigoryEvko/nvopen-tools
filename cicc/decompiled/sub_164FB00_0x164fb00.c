// Function: sub_164FB00
// Address: 0x164fb00
//
void __fastcall sub_164FB00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  int v6; // ebx
  __int64 v7; // rdi
  __int64 v8; // rax
  unsigned int v9; // eax
  const char *v10; // rax
  __int64 v11; // r15
  _BYTE *v12; // rax
  __int64 v13; // r8
  _BYTE *v14; // rax
  unsigned __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+18h] [rbp-58h]
  const char *v19; // [rsp+20h] [rbp-50h] BYREF
  char v20; // [rsp+30h] [rbp-40h]
  char v21; // [rsp+31h] [rbp-3Fh]

  v3 = a2;
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 56);
  while ( 2 )
  {
    switch ( *(_BYTE *)(a2 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v16 = *(_QWORD *)(a2 + 32);
        a2 = *(_QWORD *)(a2 + 24);
        v6 *= (_DWORD)v16;
        continue;
      case 1:
        LODWORD(v8) = 16;
        goto LABEL_4;
      case 2:
        LODWORD(v8) = 32;
        goto LABEL_4;
      case 3:
      case 9:
        LODWORD(v8) = 64;
        goto LABEL_4;
      case 4:
        LODWORD(v8) = 80;
        goto LABEL_4;
      case 5:
      case 6:
        v9 = v6 << 7;
        if ( (unsigned int)(v6 << 7) <= 7 )
          goto LABEL_8;
        goto LABEL_5;
      case 7:
        LODWORD(v8) = 8 * sub_15A9520(v7, 0);
        goto LABEL_4;
      case 0xB:
        LODWORD(v8) = *(_DWORD *)(a2 + 8) >> 8;
        goto LABEL_4;
      case 0xD:
        v8 = 8LL * *(_QWORD *)sub_15A9930(v7, a2);
        goto LABEL_4;
      case 0xE:
        v17 = *(_QWORD *)(a2 + 24);
        v18 = *(_QWORD *)(a2 + 32);
        v15 = (unsigned int)sub_15A9FE0(v7, v17);
        v8 = 8 * v18 * v15 * ((v15 + ((unsigned __int64)(sub_127FA20(v7, v17) + 7) >> 3) - 1) / v15);
        goto LABEL_4;
      case 0xF:
        LODWORD(v8) = 8 * sub_15A9520(v7, *(_DWORD *)(a2 + 8) >> 8);
LABEL_4:
        v9 = v6 * v8;
        if ( v9 <= 7 )
        {
LABEL_8:
          v21 = 1;
          v10 = "atomic memory access' size must be byte-sized";
        }
        else
        {
LABEL_5:
          if ( (v9 & (v9 - 1)) == 0 )
            return;
          v21 = 1;
          v10 = "atomic memory access' operand must have a power-of-two size";
        }
        v11 = *(_QWORD *)a1;
        v19 = v10;
        v20 = 3;
        if ( v11 )
        {
          sub_16E2CE0(&v19, v11);
          v12 = *(_BYTE **)(v11 + 24);
          if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 16) )
          {
            sub_16E7DE0(v11, 10);
          }
          else
          {
            *(_QWORD *)(v11 + 24) = v12 + 1;
            *v12 = 10;
          }
          v13 = *(_QWORD *)a1;
          *(_BYTE *)(a1 + 72) = 1;
          if ( v13 )
          {
            if ( v3 )
            {
              v14 = *(_BYTE **)(v13 + 24);
              if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 16) )
              {
                v13 = sub_16E7DE0(v13, 32);
              }
              else
              {
                *(_QWORD *)(v13 + 24) = v14 + 1;
                *v14 = 32;
              }
              sub_154E060(v3, v13, 0, 0);
            }
            sub_164FA80((__int64 *)a1, a3);
          }
        }
        else
        {
          *(_BYTE *)(a1 + 72) = 1;
        }
        return;
    }
  }
}
