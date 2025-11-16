// Function: sub_39C6FD0
// Address: 0x39c6fd0
//
unsigned __int64 __fastcall sub_39C6FD0(int *a1, __int64 a2)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rbx
  __int16 v4; // ax
  __int64 v5; // rbx
  int *v7; // rax
  size_t v8; // rdx
  int v9[8]; // [rsp+Fh] [rbp-21h] BYREF

  sub_39C2ED0(a1, 0x44u);
  sub_39C2ED0(a1, *(unsigned __int16 *)(a2 + 28));
  sub_39C6E40((__int64)a1, a2);
  v2 = *(_QWORD **)(a2 + 32);
  if ( v2 )
  {
    v3 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v3 )
    {
      while ( 2 )
      {
        v4 = *(_WORD *)(v3 + 28);
        switch ( v4 )
        {
          case 1:
          case 2:
          case 4:
          case 15:
          case 16:
          case 18:
          case 19:
          case 21:
          case 22:
          case 23:
          case 31:
          case 32:
          case 33:
          case 36:
          case 38:
          case 41:
          case 45:
          case 53:
          case 56:
          case 66:
            goto LABEL_9;
          default:
            if ( v4 != 46 )
              goto LABEL_5;
LABEL_9:
            v7 = (int *)sub_39C2E60(v3);
            if ( v8 )
              sub_39C5C30(a1, v3, v7, v8);
            else
LABEL_5:
              sub_39C6FD0(a1, v3);
            v5 = *(_QWORD *)v3;
            if ( (v5 & 4) != 0 )
              break;
            v3 = v5 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v3 )
              break;
            continue;
        }
        break;
      }
    }
  }
  LOBYTE(v9[0]) = 0;
  return sub_16C1870(a1, v9, 1u);
}
