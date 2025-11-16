// Function: sub_27DB9F0
// Address: 0x27db9f0
//
__int64 __fastcall sub_27DB9F0(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v5; // rdi
  char *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // r14
  _QWORD *v12; // [rsp+8h] [rbp-48h]
  unsigned __int8 v13; // [rsp+17h] [rbp-39h]
  _QWORD *v14; // [rsp+18h] [rbp-38h]

  v13 = 0;
  if ( a3 == *(_QWORD *)(a1 + 40) )
    v13 = (unsigned int)sub_F571B0(a1, (__int64)a2) != 0;
  v12 = (_QWORD *)(a3 + 48);
  v14 = (_QWORD *)(*(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v14 != (_QWORD *)(a3 + 48) )
  {
    do
    {
      if ( !v14 )
        BUG();
      v5 = v14[5];
      v6 = (char *)(v14 - 3);
      if ( v5 )
      {
        v7 = sub_B14240(v5);
        v9 = v8;
        v10 = v7;
        if ( v8 != v7 )
        {
          while ( *(_BYTE *)(v10 + 32) )
          {
            v10 = *(_QWORD *)(v10 + 8);
            if ( v8 == v10 )
              goto LABEL_14;
          }
LABEL_10:
          if ( v9 != v10 )
          {
            sub_B13360(v10, (unsigned __int8 *)a1, a2, 1);
            while ( 1 )
            {
              v10 = *(_QWORD *)(v10 + 8);
              if ( v9 == v10 )
                break;
              if ( !*(_BYTE *)(v10 + 32) )
                goto LABEL_10;
            }
          }
        }
      }
LABEL_14:
      if ( (char *)a1 == v6 )
        break;
      if ( !(unsigned __int8)sub_98CD80(v6) )
        break;
      v13 |= sub_BD2ED0((__int64)v6, a1, (__int64)a2);
      v14 = (_QWORD *)(*v14 & 0xFFFFFFFFFFFFFFF8LL);
    }
    while ( v12 != v14 );
  }
  if ( !*(_QWORD *)(a1 + 16) && !(unsigned __int8)sub_B46970((unsigned __int8 *)a1) )
  {
    sub_B43D60((_QWORD *)a1);
    return 1;
  }
  return v13;
}
