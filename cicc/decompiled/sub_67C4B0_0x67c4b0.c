// Function: sub_67C4B0
// Address: 0x67c4b0
//
void __fastcall sub_67C4B0(int *a1, char *a2, _DWORD *a3)
{
  __int64 v3; // r14
  char v5; // r15
  void **v6; // rcx
  __int64 v7; // rax
  _BYTE *v8; // rax
  int v9; // esi
  unsigned __int16 v10; // dx
  char v11; // dl
  __int64 v12; // rdx
  int v13; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int16 v14; // [rsp+4h] [rbp-4Ch]
  __int16 v15; // [rsp+8h] [rbp-48h]
  int v16; // [rsp+10h] [rbp-40h]

  v3 = (int)a1;
  v5 = byte_4CFFE80[4 * (int)a1 + 1];
  if ( (byte_4CFFE80[4 * (int)a1 + 2] & 4) == 0 )
    goto LABEL_6;
  if ( !*a3 )
    a3 = dword_4F07508;
  v6 = (void **)qword_4CFFD80;
  if ( !qword_4CFFD80 )
    goto LABEL_6;
  v7 = *(_QWORD *)(qword_4CFFD80 + 16);
  if ( !v7 )
    goto LABEL_6;
  v8 = (_BYTE *)(*(_QWORD *)qword_4CFFD80 + 8 * (3 * v7 - 3));
  v9 = *a3;
  if ( *(_DWORD *)v8 < *a3 || (v10 = *((_WORD *)a3 + 2), *(_DWORD *)v8 == v9) && *((_WORD *)v8 + 2) < v10 )
  {
    while ( 1 )
    {
LABEL_18:
      while ( 1 )
      {
        v11 = v8[8];
        if ( v11 == 36 )
          break;
        if ( (_DWORD)v3 == *((_DWORD *)v8 + 4) )
        {
          switch ( v11 )
          {
            case 30:
              v5 = 3;
              goto LABEL_9;
            case 31:
              v5 = 4;
              goto LABEL_9;
            case 32:
              v5 = 5;
              goto LABEL_9;
            case 33:
              v5 = 7;
              goto LABEL_9;
            case 35:
              v5 = byte_4CFFE80[4 * v3];
              goto LABEL_6;
            default:
              sub_721090(a1);
          }
        }
LABEL_16:
        if ( v8 == *v6 )
          goto LABEL_6;
        v8 -= 24;
      }
      if ( (v8[9] & 1) == 0 )
        goto LABEL_16;
      v12 = *((_QWORD *)v8 + 2);
      if ( v12 == -1 )
        goto LABEL_16;
      v8 = (char *)*v6 + 24 * v12;
    }
  }
  a1 = &v13;
  v15 &= 0xFE00u;
  v13 = v9;
  v14 = v10;
  v16 = 0;
  v8 = sub_67B720(&v13);
  if ( v8 )
  {
    v6 = (void **)qword_4CFFD80;
    goto LABEL_18;
  }
LABEL_6:
  if ( v5 )
LABEL_9:
    *a2 = v5;
}
