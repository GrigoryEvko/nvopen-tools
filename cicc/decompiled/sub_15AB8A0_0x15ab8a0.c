// Function: sub_15AB8A0
// Address: 0x15ab8a0
//
void __fastcall sub_15AB8A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // rdx
  __int64 *v6; // r13
  __int64 v7; // r15
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 *v11; // r13
  __int64 v12; // rsi
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned __int8 **v15; // r14
  unsigned __int8 *v16; // rsi
  unsigned __int8 v17; // al
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // r12
  unsigned __int8 *v21; // rsi
  unsigned __int8 v22; // al

  if ( !(unsigned __int8)sub_15AB320(a1, a2) )
    return;
  v3 = *(unsigned int *)(a2 + 8);
  v4 = *(_QWORD *)(a2 + 8 * (6 - v3));
  if ( v4 )
  {
    v5 = 8LL * *(unsigned int *)(v4 + 8);
    v6 = (__int64 *)(v4 - v5);
    if ( v4 != v4 - v5 )
    {
      do
      {
        v7 = *v6;
        if ( (unsigned __int8)sub_15AB430(a1, *v6) )
        {
          v8 = *(_QWORD *)(v7 - 8LL * *(unsigned int *)(v7 + 8));
          sub_15AB790(a1, *(unsigned __int8 **)(v8 - 8LL * *(unsigned int *)(v8 + 8)));
          sub_15ABBA0(a1, *(_QWORD *)(v8 + 8 * (3LL - *(unsigned int *)(v8 + 8))));
        }
        ++v6;
      }
      while ( (__int64 *)v4 != v6 );
      v3 = *(unsigned int *)(a2 + 8);
    }
  }
  v9 = *(_QWORD *)(a2 + 8 * (4 - v3));
  if ( v9 )
  {
    v10 = 8LL * *(unsigned int *)(v9 + 8);
    v11 = (__int64 *)(v9 - v10);
    if ( v9 != v9 - v10 )
    {
      do
      {
        v12 = *v11++;
        sub_15ABBA0(a1, v12);
      }
      while ( (__int64 *)v9 != v11 );
      v3 = *(unsigned int *)(a2 + 8);
    }
  }
  v13 = *(_QWORD *)(a2 + 8 * (5 - v3));
  if ( v13 )
  {
    v14 = 8LL * *(unsigned int *)(v13 + 8);
    v15 = (unsigned __int8 **)(v13 - v14);
    if ( v13 != v13 - v14 )
    {
      while ( 1 )
      {
        v16 = *v15;
        v17 = **v15;
        if ( v17 <= 0xEu )
        {
          if ( v17 <= 0xAu )
            goto LABEL_16;
LABEL_19:
          ++v15;
          sub_15ABBA0(a1, v16);
          if ( (unsigned __int8 **)v13 == v15 )
          {
LABEL_20:
            v3 = *(unsigned int *)(a2 + 8);
            break;
          }
        }
        else
        {
          if ( (unsigned __int8)(v17 - 32) <= 1u )
            goto LABEL_19;
LABEL_16:
          ++v15;
          sub_15ABAC0(a1);
          if ( (unsigned __int8 **)v13 == v15 )
            goto LABEL_20;
        }
      }
    }
  }
  v18 = *(_QWORD *)(a2 + 8 * (7 - v3));
  if ( v18 )
  {
    v19 = 8LL * *(unsigned int *)(v18 + 8);
    v20 = v18 - v19;
    if ( v18 != v18 - v19 )
    {
      while ( 1 )
      {
        v21 = *(unsigned __int8 **)(*(_QWORD *)v20 + 8 * (1LL - *(unsigned int *)(*(_QWORD *)v20 + 8LL)));
        v22 = *v21;
        if ( *v21 <= 0xEu )
        {
          if ( v22 <= 0xAu )
            goto LABEL_25;
LABEL_28:
          v20 += 8;
          sub_15ABBA0(a1, v21);
          if ( v18 == v20 )
            return;
        }
        else
        {
          if ( (unsigned __int8)(v22 - 32) <= 1u )
            goto LABEL_28;
          switch ( v22 )
          {
            case 0x11u:
              sub_15ABAC0(a1);
              break;
            case 0x14u:
              sub_15AB790(a1, *(unsigned __int8 **)&v21[8 * (1LL - *((unsigned int *)v21 + 2))]);
              break;
            case 0x15u:
              sub_15AB790(a1, *(unsigned __int8 **)&v21[-8 * *((unsigned int *)v21 + 2)]);
              break;
          }
LABEL_25:
          v20 += 8;
          if ( v18 == v20 )
            return;
        }
      }
    }
  }
}
