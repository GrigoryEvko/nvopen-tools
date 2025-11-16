// Function: sub_3574C70
// Address: 0x3574c70
//
void __fastcall sub_3574C70(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  _QWORD *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rsi
  unsigned int v8; // edx
  _QWORD *v9; // rdi
  _QWORD *v10; // r8
  int v11; // edi
  __int64 i; // rbx
  int v13; // r10d

  v2 = a2 + 320;
  v3 = *(_QWORD *)(a2 + 328);
  if ( v3 != a2 + 320 )
  {
    do
    {
      v5 = sub_2E5FC60(*(_QWORD *)(a1 + 224), v3);
      if ( v5 )
      {
        v6 = *(unsigned int *)(a1 + 1568);
        v7 = *(_QWORD *)(a1 + 1552);
        if ( !(_DWORD)v6 )
          goto LABEL_10;
        v8 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v9 = (_QWORD *)(v7 + 8LL * v8);
        v10 = (_QWORD *)*v9;
        if ( v5 != (_QWORD *)*v9 )
        {
          v11 = 1;
          while ( v10 != (_QWORD *)-4096LL )
          {
            v13 = v11 + 1;
            v8 = (v6 - 1) & (v11 + v8);
            v9 = (_QWORD *)(v7 + 8LL * v8);
            v10 = (_QWORD *)*v9;
            if ( v5 == (_QWORD *)*v9 )
              goto LABEL_5;
            v11 = v13;
          }
LABEL_10:
          for ( i = *(_QWORD *)(v3 + 56); v3 + 48 != i; i = *(_QWORD *)(i + 8) )
          {
            if ( (unsigned __int8)sub_3574BC0((_QWORD *)a1, i) )
              break;
            if ( !i )
              BUG();
            if ( (*(_BYTE *)i & 4) == 0 && (*(_BYTE *)(i + 44) & 8) != 0 )
            {
              do
                i = *(_QWORD *)(i + 8);
              while ( (*(_BYTE *)(i + 44) & 8) != 0 );
            }
          }
          goto LABEL_6;
        }
LABEL_5:
        if ( v9 == (_QWORD *)(v7 + 8 * v6) )
          goto LABEL_10;
      }
LABEL_6:
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v2 != v3 );
  }
}
