// Function: sub_1360E90
// Address: 0x1360e90
//
__int64 __fastcall sub_1360E90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdx
  int v5; // ecx
  _QWORD *v7; // rax
  _QWORD *v8; // r12
  __int64 v9; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // rdi
  _QWORD *v12; // rax

  result = 0;
  if ( a2 == a3 )
  {
    result = 1;
    if ( *(_BYTE *)(a2 + 16) > 0x17u )
    {
      v4 = *(unsigned int *)(a1 + 812);
      v5 = *(_DWORD *)(a1 + 816);
      if ( v5 != (_DWORD)v4 )
      {
        result = 0;
        if ( (unsigned int)(v4 - v5) <= 0x14 )
        {
          v7 = *(_QWORD **)(a1 + 800);
          if ( v7 != *(_QWORD **)(a1 + 792) )
            v4 = *(unsigned int *)(a1 + 808);
          v8 = &v7[v4];
          if ( v7 != v8 )
          {
            while ( 1 )
            {
              v9 = *v7;
              v10 = v7;
              if ( *v7 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v8 == ++v7 )
                return 1;
            }
            if ( v7 != v8 )
            {
              while ( 1 )
              {
                v11 = *(_QWORD *)(v9 + 48);
                if ( v11 )
                  v11 -= 24;
                if ( (unsigned __int8)sub_137E580(v11, a2, *(_QWORD *)(a1 + 40), *(_QWORD *)(a1 + 48)) )
                  return 0;
                v12 = v10 + 1;
                if ( v10 + 1 == v8 )
                  return 1;
                v9 = *v12;
                ++v10;
                if ( *v12 >= 0xFFFFFFFFFFFFFFFELL )
                  break;
LABEL_20:
                if ( v8 == v10 )
                  return 1;
              }
              while ( v8 != ++v12 )
              {
                v9 = *v12;
                v10 = v12;
                if ( *v12 < 0xFFFFFFFFFFFFFFFELL )
                  goto LABEL_20;
              }
            }
          }
          return 1;
        }
      }
    }
  }
  return result;
}
