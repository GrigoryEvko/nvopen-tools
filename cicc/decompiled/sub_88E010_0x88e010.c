// Function: sub_88E010
// Address: 0x88e010
//
void __fastcall sub_88E010(__int64 a1, int a2, __int64 *a3)
{
  char v3; // al
  const char *v5; // r15
  __int64 **v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 *v9[7]; // [rsp+8h] [rbp-38h] BYREF

  v9[0] = a3;
  if ( a3 )
  {
    v3 = *((_BYTE *)a3 + 8);
    if ( v3 == 3 )
    {
      sub_72F220(v9);
      a3 = v9[0];
      if ( !v9[0] )
        return;
      v3 = *((_BYTE *)v9[0] + 8);
    }
    v5 = "__global__ function";
    if ( !a2 )
      v5 = "device variable";
LABEL_5:
    if ( v3 )
    {
LABEL_6:
      if ( v3 == 2 )
      {
        v8 = a3[4];
        if ( v8 )
        {
          if ( (*(_BYTE *)(v8 + 89) & 4) != 0 && (unsigned __int8)((*(_BYTE *)(v8 + 88) & 3) - 1) <= 1u )
            sub_684AA0(7u, a2 == 0 ? 3640 : 3587, dword_4F07508);
        }
      }
      v6 = (__int64 **)v9[0];
      if ( *((_BYTE *)v9[0] + 8) == 1 && (v9[0][3] & 1) == 0 )
      {
        if ( (unsigned int)sub_8D2660(*(_QWORD *)(v9[0][4] + 128))
          || (unsigned int)sub_8D2E30(*(_QWORD *)(v9[0][4] + 128)) )
        {
          sub_6849F0(4u, 0xE00u, dword_4F07508, (__int64)v5);
        }
        v6 = (__int64 **)v9[0];
      }
    }
    else
    {
      while ( 1 )
      {
        v7 = a3[4];
        dword_4F60184 = a2;
        dword_4F60180 = 0;
        sub_8D9600(v7, sub_88E220, 794);
        v6 = (__int64 **)v9[0];
        if ( !dword_4F60180 )
          break;
        if ( !a2 )
        {
          *(_BYTE *)(a1 + 156) |= 0x20u;
          break;
        }
        if ( unk_4D0452C )
          break;
        *(_BYTE *)(a1 + 198) |= 4u;
        a3 = *v6;
        v9[0] = a3;
        if ( !a3 )
          return;
LABEL_9:
        v3 = *((_BYTE *)a3 + 8);
        if ( v3 != 3 )
          goto LABEL_5;
        sub_72F220(v9);
        a3 = v9[0];
        if ( !v9[0] )
          return;
        v3 = *((_BYTE *)v9[0] + 8);
        if ( v3 )
          goto LABEL_6;
      }
    }
    a3 = *v6;
    v9[0] = a3;
    if ( a3 )
      goto LABEL_9;
  }
}
