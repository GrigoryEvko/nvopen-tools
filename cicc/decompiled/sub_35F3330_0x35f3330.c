// Function: sub_35F3330
// Address: 0x35f3330
//
__int64 __fastcall sub_35F3330(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  void *v7; // rcx
  unsigned __int64 v8; // rdx
  unsigned int v9; // edx
  unsigned int v10; // eax
  void *v11; // rdx
  unsigned int v12; // eax
  _QWORD *v13; // rdx
  void *v14; // rdx
  __int64 v15; // rdx
  _QWORD *v16; // rdx
  void *v17; // rdx
  void *v18; // rdx
  void *v19; // rdx
  void *v20; // rdx
  __int64 v21; // rdx
  _QWORD *v22; // rdx
  void *v23; // rdx
  void *v24; // rdx

  result = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( !strcmp((const char *)a5, "kind") )
  {
    switch ( ((unsigned int)result >> 6) & 7 )
    {
      case 0u:
        v14 = *(void **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v14 <= 0xDu )
          return sub_CB6200(a4, "kind::mxf4nvf4", 0xEu);
        qmemcpy(v14, "kind::mxf4nvf4", 14);
        *(_QWORD *)(a4 + 32) += 14LL;
        return 13414;
      case 1u:
        v19 = *(void **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v19 <= 0xBu )
          return sub_CB6200(a4, "kind::f8f6f4", 0xCu);
        qmemcpy(v19, "kind::f8f6f4", 12);
        *(_QWORD *)(a4 + 32) += 12LL;
        return 0x38663A3A646E696BLL;
      case 2u:
        v20 = *(void **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v20 <= 0xDu )
          return sub_CB6200(a4, "kind::mxf8f6f4", 0xEu);
        qmemcpy(v20, "kind::mxf8f6f4", 14);
        *(_QWORD *)(a4 + 32) += 14LL;
        return 0x786D3A3A646E696BLL;
      case 3u:
        v15 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v15) <= 8 )
          return sub_CB6200(a4, "kind::f16", 9u);
        *(_BYTE *)(v15 + 8) = 54;
        *(_QWORD *)v15 = 0x31663A3A646E696BLL;
        *(_QWORD *)(a4 + 32) += 9LL;
        return 0x31663A3A646E696BLL;
      case 4u:
        v16 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v16 <= 7u )
          return sub_CB6200(a4, "kind::i8", 8u);
        *v16 = 0x38693A3A646E696BLL;
        *(_QWORD *)(a4 + 32) += 8LL;
        return 0x38693A3A646E696BLL;
      case 5u:
        v17 = *(void **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v17 <= 9u )
          return sub_CB6200(a4, "kind::tf32", 0xAu);
        qmemcpy(v17, "kind::tf32", 10);
        *(_QWORD *)(a4 + 32) += 10LL;
        return 0x66743A3A646E696BLL;
      case 6u:
        goto LABEL_64;
      case 7u:
        v18 = *(void **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v18 <= 9u )
          return sub_CB6200(a4, "kind::mxf4", 0xAu);
        qmemcpy(v18, "kind::mxf4", 10);
        *(_QWORD *)(a4 + 32) += 10LL;
        return 0x786D3A3A646E696BLL;
    }
  }
  if ( !strcmp((const char *)a5, "cta_group") )
  {
    v7 = *(void **)(a4 + 32);
    v8 = *(_QWORD *)(a4 + 24) - (_QWORD)v7;
    if ( (result & 2) != 0 )
    {
      if ( v8 <= 0xB )
      {
        return sub_CB6200(a4, (unsigned __int8 *)"cta_group::2", 0xCu);
      }
      else
      {
        qmemcpy(v7, "cta_group::2", 12);
        *(_QWORD *)(a4 + 32) += 12LL;
        return 0x756F72675F617463LL;
      }
    }
    else if ( v8 <= 0xB )
    {
      return sub_CB6200(a4, (unsigned __int8 *)"cta_group::1", 0xCu);
    }
    else
    {
      qmemcpy(v7, "cta_group::1", 12);
      *(_QWORD *)(a4 + 32) += 12LL;
      return 0x756F72675F617463LL;
    }
  }
  if ( !strcmp((const char *)a5, "scale") )
  {
    v9 = ((unsigned int)result >> 2) & 3;
    if ( (((unsigned int)result >> 2) & 3) == 2 )
      goto LABEL_13;
    if ( (_BYTE)v9 == 3 )
    {
      v24 = *(void **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v24 <= 0xDu )
      {
        return sub_CB6200(a4, ".scale_vec::4X", 0xEu);
      }
      else
      {
        qmemcpy(v24, ".scale_vec::4X", 14);
        *(_QWORD *)(a4 + 32) += 14LL;
        return 0x765F656C6163732ELL;
      }
    }
    else
    {
      if ( !(_BYTE)v9 )
      {
        v10 = ((unsigned int)result >> 6) & 7;
        if ( (_BYTE)v10 != 2 )
        {
          if ( (_BYTE)v10 == 7 )
          {
LABEL_13:
            v11 = *(void **)(a4 + 32);
            if ( *(_QWORD *)(a4 + 24) - (_QWORD)v11 <= 0xDu )
              return sub_CB6200(a4, ".scale_vec::2X", 0xEu);
            qmemcpy(v11, ".scale_vec::2X", 14);
            *(_QWORD *)(a4 + 32) += 14LL;
            return 0x765F656C6163732ELL;
          }
LABEL_64:
          BUG();
        }
      }
      v23 = *(void **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v23 <= 0xDu )
      {
        return sub_CB6200(a4, ".scale_vec::1X", 0xEu);
      }
      else
      {
        qmemcpy(v23, ".scale_vec::1X", 14);
        *(_QWORD *)(a4 + 32) += 14LL;
        return 0x765F656C6163732ELL;
      }
    }
  }
  else if ( !strcmp((const char *)a5, "alias_scale") )
  {
    v12 = ((unsigned int)result >> 9) & 3;
    if ( v12 )
    {
      if ( (_BYTE)v12 != 1 )
        goto LABEL_64;
      v13 = *(_QWORD **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v13 <= 7u )
      {
        return sub_CB6200(a4, ".block32", 8u);
      }
      else
      {
        *v13 = 0x32336B636F6C622ELL;
        *(_QWORD *)(a4 + 32) += 8LL;
        return 0x32336B636F6C622ELL;
      }
    }
    else
    {
      v22 = *(_QWORD **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v22 <= 7u )
      {
        return sub_CB6200(a4, ".block16", 8u);
      }
      else
      {
        *v22 = 0x36316B636F6C622ELL;
        *(_QWORD *)(a4 + 32) += 8LL;
        return 0x36316B636F6C622ELL;
      }
    }
  }
  else if ( *(_BYTE *)a5 == 119 && *(_BYTE *)(a5 + 1) == 115 && !*(_BYTE *)(a5 + 2) && (result & 1) != 0 )
  {
    v21 = *(_QWORD *)(a4 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v21) <= 2 )
    {
      return sub_CB6200(a4, ".ws", 3u);
    }
    else
    {
      result = 30510;
      *(_BYTE *)(v21 + 2) = 115;
      *(_WORD *)v21 = 30510;
      *(_QWORD *)(a4 + 32) += 3LL;
    }
  }
  return result;
}
