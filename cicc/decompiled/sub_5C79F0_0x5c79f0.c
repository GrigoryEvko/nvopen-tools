// Function: sub_5C79F0
// Address: 0x5c79f0
//
char *__fastcall sub_5C79F0(__int64 a1)
{
  const char *v1; // rdx
  char *v2; // r8
  int v3; // eax

  v1 = *(const char **)(a1 + 24);
  v2 = *(char **)(a1 + 16);
  if ( v1 )
  {
    v3 = sprintf(byte_4CF79C0, "%s::%s", v1, *(const char **)(a1 + 16));
    v2 = (char *)sub_7248C0(0, byte_4CF79C0, v3);
  }
  switch ( *(_BYTE *)(a1 + 8) )
  {
    case 'V':
      v2 = "__host__";
      break;
    case 'W':
      v2 = "__device__";
      break;
    case 'X':
      v2 = "__global__";
      break;
    case 'Y':
      v2 = "__tile_global__";
      break;
    case 'Z':
      v2 = "__shared__";
      break;
    case '[':
      v2 = "__constant__";
      break;
    case '\\':
      v2 = "__launch_bounds__";
      break;
    case ']':
      v2 = "__maxnreg__";
      break;
    case '^':
      v2 = "__local_maxnreg__";
      break;
    case '_':
      v2 = "__tile_builtin__";
      break;
    case 'f':
      v2 = "__managed__";
      break;
    case 'k':
      v2 = "__cluster_dims__";
      break;
    case 'l':
      v2 = "__block_size__";
      break;
    case 'r':
      v2 = "__nv_pure__";
      break;
    default:
      if ( !v2 )
        v2 = (char *)byte_3F871B3;
      break;
  }
  return v2;
}
