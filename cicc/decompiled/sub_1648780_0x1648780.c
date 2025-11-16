// Function: sub_1648780
// Address: 0x1648780
//
void __fastcall sub_1648780(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // r9d
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // r11
  _QWORD *v8; // rdx
  __int64 v9; // rbx
  unsigned __int64 v10; // r8
  __int64 v11; // r8

  if ( a2 != a3 )
  {
    v3 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    if ( v3 )
    {
      v5 = 0;
      v6 = 0;
      v7 = a3 + 8;
      while ( 1 )
      {
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        {
          v8 = (_QWORD *)(v5 + *(_QWORD *)(a1 - 8));
          if ( a2 != *v8 )
            goto LABEL_5;
LABEL_8:
          if ( a2 )
          {
            v9 = v8[1];
            v10 = v8[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v10 = v9;
            if ( v9 )
              *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
          }
          *v8 = a3;
          if ( !a3 )
            goto LABEL_5;
          v11 = *(_QWORD *)(a3 + 8);
          v8[1] = v11;
          if ( v11 )
            *(_QWORD *)(v11 + 16) = (unsigned __int64)(v8 + 1) | *(_QWORD *)(v11 + 16) & 3LL;
          ++v6;
          v5 += 24;
          v8[2] = v7 | v8[2] & 3LL;
          *(_QWORD *)(a3 + 8) = v8;
          if ( v3 == (_DWORD)v6 )
            return;
        }
        else
        {
          v8 = (_QWORD *)(a1 + 24 * (v6 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
          if ( a2 == *v8 )
            goto LABEL_8;
LABEL_5:
          ++v6;
          v5 += 24;
          if ( v3 == (_DWORD)v6 )
            return;
        }
      }
    }
  }
}
