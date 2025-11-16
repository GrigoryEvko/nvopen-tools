// Function: sub_30275E0
// Address: 0x30275e0
//
void __fastcall sub_30275E0(__int64 a1, _BYTE *a2, __int64 a3)
{
  char v4; // al
  _BYTE *v5; // rbx
  unsigned int v6; // eax
  _BYTE *v7; // rax
  signed __int64 v8; // rsi
  _BYTE *v9; // rsi
  __int64 v10; // rax
  _BYTE *v11; // rax
  _BYTE *v12; // rax
  _BYTE *v13; // rax
  _BYTE *v14; // rax
  _BYTE *v15; // rax
  _BYTE *v16; // rax

  v4 = *a2;
  v5 = a2;
  while ( 2 )
  {
    switch ( v4 )
    {
      case 0:
        v9 = (_BYTE *)*((_QWORD *)v5 + 2);
        if ( (unsigned __int8)(*v9 - 1) <= 1u || *v9 == 4 )
        {
          sub_30275E0(a1, v9, a3);
        }
        else
        {
          v14 = *(_BYTE **)(a3 + 32);
          if ( (unsigned __int64)v14 >= *(_QWORD *)(a3 + 24) )
          {
            sub_CB5D20(a3, 40);
          }
          else
          {
            *(_QWORD *)(a3 + 32) = v14 + 1;
            *v14 = 40;
          }
          sub_30275E0(a1, *((_QWORD *)v5 + 2), a3);
          v15 = *(_BYTE **)(a3 + 32);
          if ( (unsigned __int64)v15 >= *(_QWORD *)(a3 + 24) )
          {
            sub_CB5D20(a3, 41);
          }
          else
          {
            *(_QWORD *)(a3 + 32) = v15 + 1;
            *v15 = 41;
          }
        }
        if ( *(_DWORD *)v5 >> 8 )
          goto LABEL_42;
        v10 = *((_QWORD *)v5 + 3);
        if ( *(_BYTE *)v10 == 1 )
        {
          v8 = *(_QWORD *)(v10 + 16);
          if ( v8 < 0 )
          {
LABEL_11:
            sub_CB59F0(a3, v8);
            return;
          }
        }
        v11 = *(_BYTE **)(a3 + 32);
        if ( (unsigned __int64)v11 >= *(_QWORD *)(a3 + 24) )
        {
          sub_CB5D20(a3, 43);
        }
        else
        {
          *(_QWORD *)(a3 + 32) = v11 + 1;
          *v11 = 43;
        }
        v4 = **((_BYTE **)v5 + 3);
        if ( (unsigned __int8)(v4 - 1) <= 1u )
        {
          v5 = (_BYTE *)*((_QWORD *)v5 + 3);
          continue;
        }
        sub_A51310(a3, 0x28u);
        sub_30275E0(a1, *((_QWORD *)v5 + 3), a3);
        sub_A51310(a3, 0x29u);
        return;
      case 1:
        v8 = *((_QWORD *)v5 + 2);
        goto LABEL_11;
      case 2:
        sub_EA12C0(*((_QWORD *)v5 + 2), a3, *(_BYTE **)(a1 + 208));
        return;
      case 3:
        v6 = *(_DWORD *)v5 >> 8;
        if ( v6 == 2 )
        {
          v16 = *(_BYTE **)(a3 + 32);
          if ( (unsigned __int64)v16 >= *(_QWORD *)(a3 + 24) )
          {
            sub_CB5D20(a3, 126);
          }
          else
          {
            *(_QWORD *)(a3 + 32) = v16 + 1;
            *v16 = 126;
          }
        }
        else if ( v6 > 2 )
        {
          if ( v6 == 3 )
          {
            v12 = *(_BYTE **)(a3 + 32);
            if ( (unsigned __int64)v12 >= *(_QWORD *)(a3 + 24) )
            {
              sub_CB5D20(a3, 43);
            }
            else
            {
              *(_QWORD *)(a3 + 32) = v12 + 1;
              *v12 = 43;
            }
          }
        }
        else if ( v6 )
        {
          v7 = *(_BYTE **)(a3 + 32);
          if ( (unsigned __int64)v7 >= *(_QWORD *)(a3 + 24) )
          {
            sub_CB5D20(a3, 45);
          }
          else
          {
            *(_QWORD *)(a3 + 32) = v7 + 1;
            *v7 = 45;
          }
        }
        else
        {
          v13 = *(_BYTE **)(a3 + 32);
          if ( (unsigned __int64)v13 >= *(_QWORD *)(a3 + 24) )
          {
            sub_CB5D20(a3, 33);
          }
          else
          {
            *(_QWORD *)(a3 + 32) = v13 + 1;
            *v13 = 33;
          }
        }
        v5 = (_BYTE *)*((_QWORD *)v5 + 2);
        v4 = *v5;
        continue;
      case 4:
        (*(void (__fastcall **)(_BYTE *, __int64, _QWORD))(*((_QWORD *)v5 - 1) + 24LL))(
          v5 - 8,
          a3,
          *(_QWORD *)(a1 + 208));
        return;
      default:
LABEL_42:
        BUG();
    }
  }
}
