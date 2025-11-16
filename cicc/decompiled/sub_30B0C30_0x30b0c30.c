// Function: sub_30B0C30
// Address: 0x30b0c30
//
__int64 __fastcall sub_30B0C30(__int64 a1, int a2)
{
  char *v2; // r14
  size_t v4; // rax
  void *v5; // rdi
  size_t v6; // r13

  switch ( a2 )
  {
    case 0:
      v2 = "?? (error)";
      break;
    case 1:
      v2 = "single-instruction";
      break;
    case 2:
      v2 = "multi-instruction";
      break;
    case 3:
      v2 = "pi-block";
      break;
    case 4:
      v2 = "root";
      break;
    default:
      break;
  }
  v4 = strlen(v2);
  v5 = *(void **)(a1 + 32);
  v6 = v4;
  if ( v4 > *(_QWORD *)(a1 + 24) - (_QWORD)v5 )
  {
    sub_CB6200(a1, (unsigned __int8 *)v2, v4);
    return a1;
  }
  else
  {
    if ( v4 )
    {
      memcpy(v5, v2, v4);
      *(_QWORD *)(a1 + 32) += v6;
    }
    return a1;
  }
}
